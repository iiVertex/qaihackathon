# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "gliner2[local]==2.1.0",
#   "huggingface_hub==2.4.0",
#   "numpy==2.4.6",
#   "pandas==3.0.6",
#   "protobuf==7.36.2",
#   "pyarrow==26.0.0",
#   "scikit-learn==1.9.1",
#   "sentencepiece==0.2.2",
#   "torch==2.14.1",
#   "transformers==5.20.2",
# ]
# ///
"""Phase 5: score scraped reviews and build a ranked brief for each business.

  signals + AI score -> credibility (0-1, never a "fake" verdict)
  GLiNER2 yes/no questions -> violations with an evidence sentence
  severity = category weight x confidence x credible count x recency
  brief in English and Arabic, from an LLM with a labelled template fallback

Credibility and severity weights are hand-set heuristics, not calibrated values.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gliner2 import AutoExtractor
from huggingface_hub import HfApi, InferenceClient, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET_REPO = "Hermesione/reviewere-dataset"
GLINER_MODEL = "fastino/GLiNER2.5-multi-Decide"
DETECTOR_MODEL = "Hermesione/robert-fake-reviewer"
BRIEF_MODEL = "zai-org/GLM-5.3"
OUTPUT_DIR = Path("output")

QUESTIONS = {
    "hygiene": "Does the reviewer report a hygiene problem: food poisoning, pests, a dirty kitchen or room, spoiled food, or a bad smell?",
    "safety": "Does the reviewer report a safety danger: a fire hazard, blocked exit, injury, broken structure, or electrical problem?",
    "pricing_fraud": "Does the reviewer report being overcharged, hidden fees, a bill different from the menu, or a refused refund?",
    "licensing": "Does the reviewer report unlicensed activity, an alcohol or smoking violation, or an expired permit?",
    "service_only": "Is the reviewer complaining only about rude staff or slow service?",
}
SENTIMENT_QUESTION = "Is this review mostly positive?"
TASKS = {name: {"labels": ["yes", "no"], "prompt": prompt} for name, prompt in QUESTIONS.items()}
TASKS["positive"] = {"labels": ["yes", "no"], "prompt": SENTIMENT_QUESTION}

CATEGORY_WEIGHT = {"safety": 1.0, "hygiene": 0.9, "licensing": 0.7, "pricing_fraud": 0.6, "service_only": 0.1}
CATEGORY_NAMES = {
    "safety": ("safety", "السلامة"),
    "hygiene": ("hygiene", "النظافة"),
    "licensing": ("licensing", "التراخيص"),
    "pricing_fraud": ("pricing", "الأسعار"),
    "service_only": ("service", "الخدمة"),
}
NEVER_DROPPED = ("safety", "hygiene")
LOW_CONFIDENCE = 0.6
CREDIBLE = 0.5
RECENCY_HALF_LIFE_DAYS = 365
LOW_CONFIDENCE_FLOOR_COUNT = 0.5

PENALTY_SINGLE_REVIEW_AUTHOR = 0.20
PENALTY_BURST = 0.20
PENALTY_NEAR_DUPLICATE = 0.30
PENALTY_RATING_MISMATCH = 0.15
PENALTY_AI_MAX = 0.15  # Small on purpose: about 1 in 7 human test reviews were flagged.
CREDIBILITY_FLOOR = 0.05

BURST_WINDOW = pd.Timedelta(hours=48)
BURST_MIN_REVIEWS = 3
NEAR_DUPLICATE_SIMILARITY = 0.9
NEAR_DUPLICATE_MIN_CHARS = 40
AI_MAX_LENGTH = 256
MAX_LLM_CALLS = 20  # The approval limit for Inference Providers in CLAUDE.md.
BRIEF_MAX_TOKENS = 3000  # GLM-5.3 spends most of its tokens reasoning before it writes.
SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])\s+|\s*[،؛]\s*")
MIN_SENTENCE_CHARS = 8


def load_reviews(limit: int | None) -> pd.DataFrame:
    path = hf_hub_download(DATASET_REPO, "scraped_reviews.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    print(f"loaded {len(df)} reviews from {DATASET_REPO}")
    if limit:
        df = df.groupby("business", group_keys=False).head(limit)
        print(f"limited to {len(df)} reviews ({limit} per business)")
    return df.reset_index(drop=True)


def answer_of(value) -> tuple[str | None, float]:
    if isinstance(value, dict):
        return value.get("label"), float(value.get("confidence") or 0)
    return value, 0.0


def best_evidence(model, text: str, category: str) -> str:
    """The highest-scoring sentence answering yes, or the whole review if none does."""
    best, best_score = text, 0.0
    for sentence in SENTENCE_BREAK.split(text):
        if len(sentence) < MIN_SENTENCE_CHARS or sentence == text:
            continue
        raw = model.classify_text(sentence, {category: TASKS[category]}, include_confidence=True)
        label, score = answer_of(raw.get(category))
        if label == "yes" and score > best_score:
            best, best_score = sentence, score
    return best


def classify(df: pd.DataFrame) -> pd.DataFrame:
    model = AutoExtractor.from_pretrained(GLINER_MODEL)
    columns = {f"{c}_{suffix}": [] for c in QUESTIONS for suffix in ("confidence", "evidence")}
    columns["positive"] = []
    started = time.time()
    for index, text in enumerate(df["text"], 1):
        raw = model.classify_text(text, TASKS, include_confidence=True)
        columns["positive"].append(answer_of(raw.get("positive"))[0] == "yes")
        for category in QUESTIONS:
            label, score = answer_of(raw.get(category))
            hit = label == "yes"
            columns[f"{category}_confidence"].append(score if hit else np.nan)
            columns[f"{category}_evidence"].append(best_evidence(model, text, category) if hit else None)
        if index % 50 == 0 or index == len(df):
            print(f"classified {index}/{len(df)} ({(time.time() - started) / index:.2f}s per review)", flush=True)
    return df.assign(**columns)


def ai_scores(df: pd.DataFrame) -> pd.Series:
    """English only: the detector has no Arabic training data."""
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL).eval()
    english = df[df["language"] == "en"]
    scores = []
    with torch.no_grad():
        for start in range(0, len(english), 16):
            batch = tokenizer(english["text"].iloc[start:start + 16].tolist(), padding=True, truncation=True,
                              max_length=AI_MAX_LENGTH, return_tensors="pt")
            scores += torch.softmax(model(**batch).logits, dim=1)[:, 1].tolist()
    print(f"AI score computed for {len(english)} English reviews")
    return pd.Series(scores, index=english.index, dtype=float).reindex(df.index)


def burst_flags(df: pd.DataFrame) -> pd.Series:
    """5-star reviews with at least two other 5-star reviews within 48 hours.

    Only the newest sample has consecutive reviews, so only it is checked.
    """
    flags = pd.Series(False, index=df.index)
    five_star = df[(df["rating"] == 5) & (df["sample"] == "newest")]
    for _, group in five_star.groupby("business"):
        times = group["timestamp_utc"].sort_values()
        for index, moment in times.items():
            close = ((times - moment).abs() <= BURST_WINDOW).sum()
            flags[index] = close >= BURST_MIN_REVIEWS
    return flags


def near_duplicate_flags(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    long_enough = df[df["text"].str.len() >= NEAR_DUPLICATE_MIN_CHARS]
    if len(long_enough) < 2:
        return flags
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(long_enough["text"])
    similar = cosine_similarity(matrix, dense_output=False)
    authors = long_enough["author_hash"].to_numpy()
    rows, cols = (similar >= NEAR_DUPLICATE_SIMILARITY).nonzero()
    for row, col in zip(rows, cols):
        if row != col and authors[row] != authors[col]:
            flags[long_enough.index[row]] = True
    return flags


def add_credibility(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_score"] = ai_scores(df)
    df["ai_check_available"] = df["ai_score"].notna()
    df["signal_single_review_author"] = df["author_review_count"].fillna(0) <= 1
    df["signal_burst"] = burst_flags(df)
    df["signal_near_duplicate"] = near_duplicate_flags(df)
    df["signal_rating_mismatch"] = ((df["rating"] >= 4) & ~df["positive"]) | ((df["rating"] <= 2) & df["positive"])
    penalty = (
        df["signal_single_review_author"] * PENALTY_SINGLE_REVIEW_AUTHOR
        + df["signal_burst"] * PENALTY_BURST
        + df["signal_near_duplicate"] * PENALTY_NEAR_DUPLICATE
        + df["signal_rating_mismatch"] * PENALTY_RATING_MISMATCH
        + df["ai_score"].fillna(0) * PENALTY_AI_MAX
    )
    df["credibility"] = (1 - penalty).clip(lower=CREDIBILITY_FLOOR)
    for category in QUESTIONS:
        hit = df[f"{category}_confidence"].notna()
        low = (df[f"{category}_confidence"] < LOW_CONFIDENCE) | (df["credibility"] < CREDIBLE)
        df[f"{category}_low_confidence"] = hit & low
    return df


def severity_of(hits: pd.DataFrame, category: str, now: pd.Timestamp) -> tuple[float, int, int]:
    """Returns (severity, credible hits, all hits)."""
    credible = hits[hits["credibility"] >= CREDIBLE]
    if credible.empty and (hits.empty or category not in NEVER_DROPPED):
        return 0.0, 0, len(hits)
    used = credible if len(credible) else hits
    count = len(credible) if len(credible) else LOW_CONFIDENCE_FLOOR_COUNT
    ages = (now - used["timestamp_utc"]).dt.days.clip(lower=0)
    recency = (0.5 ** (ages / RECENCY_HALF_LIFE_DAYS)).mean()
    severity = CATEGORY_WEIGHT[category] * used[f"{category}_confidence"].mean() * count * recency
    return float(severity), len(credible), len(hits)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for business, group in df.groupby("business"):
        first = group.iloc[0]
        row = {
            "business": business,
            "category": first["category"],
            "place_rating": first["place_rating"],
            "place_reviews": first["place_reviews"],
            "reviews_scored": len(group),
            "reviews_newest": int((group["sample"] == "newest").sum()),
            "reviews_lowest_rating": int((group["sample"] == "lowest_rating").sum()),
            "mean_credibility": float(group["credibility"].mean()),
            "ai_check_available": int(group["ai_check_available"].sum()),
        }
        total = 0.0
        for category in QUESTIONS:
            hits = group[group[f"{category}_confidence"].notna()]
            severity, credible, count = severity_of(hits, category, now)
            row[f"{category}_severity"] = severity
            row[f"{category}_credible_hits"] = credible
            row[f"{category}_hits"] = count
            total += severity
        row["risk_score"] = total
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    return summary


def facts_for(row: pd.Series, reviews: pd.DataFrame) -> dict:
    group = reviews[reviews["business"] == row["business"]]
    findings = []
    for category in sorted(QUESTIONS, key=lambda c: row[f"{c}_severity"], reverse=True):
        if row[f"{category}_hits"] == 0 or category == "service_only":
            continue
        hits = group[group[f"{category}_confidence"].notna()].sort_values("credibility", ascending=False)
        findings.append({
            "category": category,
            "reviews_alleging": int(row[f"{category}_hits"]),
            "credible_reviews": int(row[f"{category}_credible_hits"]),
            "examples": hits[f"{category}_evidence"].head(2).str[:200].tolist(),
        })
    return {"business": row["business"], "type": row["category"], "rank": int(row["rank"]),
            "reviews_scored": int(row["reviews_scored"]), "findings": findings}


def llm_brief(client: InferenceClient, facts: dict, language: str) -> str:
    system = (
        f"You write short briefs for a food and hospitality inspector, in {language}. Use only the facts given. "
        "Write 3 to 4 sentences. Say what reviews allege, and never state a violation as confirmed. Never call a "
        "review fake. Do not name individual people. The sample deliberately over-includes low-rated reviews, so "
        "give counts, not percentages. Output only the brief."
    )
    response = client.chat_completion(
        model=BRIEF_MODEL, max_tokens=BRIEF_MAX_TOKENS, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty response")
    return text


def template_brief(facts: dict, arabic: bool) -> str:
    index = 1 if arabic else 0
    parts = [
        f"{CATEGORY_NAMES[f['category']][index]}: {f['reviews_alleging']} ({f['credible_reviews']})"
        for f in facts["findings"]
    ]
    listed = "، ".join(parts) if arabic else ", ".join(parts)
    if arabic:
        body = f"الشكاوى المزعومة: {listed}." if parts else "لا توجد شكاوى تنظيمية في المراجعات المفحوصة."
        return f"{facts['business']}: عدد المراجعات المفحوصة {facts['reviews_scored']}. {body} الأرقام بين القوسين هي المراجعات الموثوقة."
    body = f"Alleged complaints: {listed}." if parts else "No regulatory complaints in the reviews scored."
    return f"{facts['business']}: {facts['reviews_scored']} reviews scored. {body} Numbers in brackets are credible reviews."


def add_briefs(summary: pd.DataFrame, reviews: pd.DataFrame, max_calls: int) -> pd.DataFrame:
    calls = min(max_calls, MAX_LLM_CALLS)
    client = InferenceClient(provider="auto") if calls else None
    columns = {key: [] for key in ("brief_en", "brief_ar", "brief_source_en", "brief_source_ar")}
    used = 0
    for _, row in summary.iterrows():
        facts = facts_for(row, reviews)
        for key, language, arabic in (("en", "English", False), ("ar", "Modern Standard Arabic", True)):
            if used < calls:
                used += 1
                try:
                    columns[f"brief_{key}"].append(llm_brief(client, facts, language))
                    columns[f"brief_source_{key}"].append(f"llm:{BRIEF_MODEL}")
                    continue
                except Exception as error:  # Labelled below, never silent.
                    print(f"BRIEF FAILED {row['business']} [{key}]: {type(error).__name__}: {str(error)[:200]}")
                    reason = f"template: {type(error).__name__}"
            else:
                reason = "template: llm call limit"
            columns[f"brief_{key}"].append(template_brief(facts, arabic))
            columns[f"brief_source_{key}"].append(reason)
    print(f"LLM calls used: {used}")
    return summary.assign(**columns)


def show(summary: pd.DataFrame, reviews: pd.DataFrame) -> None:
    print("\n=== summary per business ===")
    wanted = ["rank", "business", "risk_score", "reviews_scored", "mean_credibility"]
    wanted += [f"{c}_hits" for c in ("safety", "hygiene", "licensing", "pricing_fraud")]
    print(summary[wanted].round(3).to_string(index=False))
    print("\n=== signals across reviews ===")
    print(reviews[[c for c in reviews if c.startswith("signal_")]].sum().to_string())
    print(f"credibility mean {reviews['credibility'].mean():.3f} | below {CREDIBLE}: {(reviews['credibility'] < CREDIBLE).sum()}")
    print("\n=== brief sources ===")
    print(pd.concat([summary["brief_source_en"], summary["brief_source_ar"]]).value_counts().to_string())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="reviews per business, for test runs")
    parser.add_argument("--max-llm-calls", type=int, default=MAX_LLM_CALLS)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    reviews = add_credibility(classify(load_reviews(args.limit)))
    summary = add_briefs(summarise(reviews), reviews, args.max_llm_calls)

    OUTPUT_DIR.mkdir(exist_ok=True)
    reviews.to_parquet(OUTPUT_DIR / "reviews.parquet", index=False)
    summary.to_parquet(OUTPUT_DIR / "businesses.parquet", index=False)
    print(f"saved {OUTPUT_DIR}: {len(reviews)} reviews, {len(summary)} businesses")
    show(summary, reviews)

    if args.no_push:
        print("\nNot pushed (--no-push).")
        return
    folder = "smoke/" if args.limit else ""
    hub = HfApi()
    for name in ("reviews.parquet", "businesses.parquet"):
        hub.upload_file(path_or_fileobj=str(OUTPUT_DIR / name), path_in_repo=f"{folder}{name}", repo_id=DATASET_REPO,
                        repo_type="dataset", commit_message=f"Phase 5: {folder}{name}")
    print(f"pushed: https://huggingface.co/datasets/{DATASET_REPO}/tree/main/{folder}")


if __name__ == "__main__":
    main()
