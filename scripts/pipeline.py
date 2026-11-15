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
  severity = category weight x confidence x credible complaints per 1,000 Google reviews x recency
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
from huggingface_hub import HfApi, InferenceClient, hf_hub_download, snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET_REPO = "Hermesione/reviewere-dataset"
GLINER_MODEL = "fastino/GLiNER2.5-multi-Decide"
DETECTOR_MODEL = "Hermesione/robert-fake-reviewer"
BRIEF_MODEL = "zai-org/GLM-5.3"
OUTPUT_DIR = Path("output")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTIONS = {
    "hygiene": "Does the reviewer complain that the place was dirty or unhygienic, such as pests, spoiled food, bad smells, or getting sick from the food?",
    "safety": "Does the reviewer describe a physical danger to people, such as a fire hazard, blocked exit, injury, collapsing structure, or exposed electrical wiring?",
    "pricing_fraud": "Does the reviewer accuse the business of overcharging, hidden fees, charging more than the menu price, or refusing a refund?",
    "licensing": "Does the reviewer report illegal or unlicensed activity, such as alcohol or smoking violations or operating without a permit?",
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
HIGH_STARS = 4
MIXED_STARS = 3

PENALTY_SINGLE_REVIEW_AUTHOR = 0.20
PENALTY_BURST = 0.20
PENALTY_NEAR_DUPLICATE = 0.30
PENALTY_RATING_MISMATCH = 0.15
PENALTY_AI_MAX = 0.15  # Small on purpose: about 1 in 7 human test reviews were flagged.
CREDIBILITY_FLOOR = 0.05

BURST_WINDOW = pd.Timedelta(hours=48)
BURST_MIN_REVIEWS = 3
BURST_FACTOR = 3  # A burst needs this many times the business's usual 5-star pace.
NEAR_DUPLICATE_SIMILARITY = 0.9
NEAR_DUPLICATE_MIN_CHARS = 40
AI_MAX_LENGTH = 256
MAX_LLM_CALLS = 20  # The approval limit for Inference Providers in CLAUDE.md.
BRIEF_MAX_TOKENS = 6000  # GLM-5.3 spends most of its tokens reasoning before it writes.
SENTENCE_BREAK = re.compile(r"(?<=[.!?؟])\s+|\s*[،؛]\s*")
MIN_SENTENCE_CHARS = 8
PROGRESS_EVERY = 64


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


def batch_evidence(model, texts: list[str], hits: list[int], category: str, batch_size: int) -> dict[int, str]:
    """Batched best_evidence for every hit in one category."""
    best = {index: (texts[index], 0.0) for index in hits}
    pairs = [(index, sentence) for index in hits for sentence in SENTENCE_BREAK.split(texts[index])
             if len(sentence) >= MIN_SENTENCE_CHARS and sentence != texts[index]]
    if pairs:
        answers = model.batch_classify_text([sentence for _, sentence in pairs], {category: TASKS[category]},
                                            batch_size=batch_size, include_confidence=True)
        for (index, sentence), answer in zip(pairs, answers, strict=True):
            label, score = answer_of(answer.get(category))
            if label == "yes" and score > best[index][1]:
                best[index] = (sentence, score)
    return {index: sentence for index, (sentence, _) in best.items()}


def classify(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    started = time.time()
    model = AutoExtractor.from_pretrained(GLINER_MODEL).to(DEVICE)
    print(f"GLiNER2 on {DEVICE}, loaded in {time.time() - started:.0f}s", flush=True)

    texts = df["text"].tolist()
    answers = []
    started = time.time()
    for start in range(0, len(texts), PROGRESS_EVERY):
        answers += model.batch_classify_text(texts[start:start + PROGRESS_EVERY], TASKS, batch_size=batch_size,
                                             include_confidence=True)
        print(f"classified {len(answers)}/{len(texts)} ({(time.time() - started) / len(answers):.3f}s per review)", flush=True)

    columns = {"positive": [answer_of(a.get("positive"))[0] == "yes" for a in answers]}
    started = time.time()
    for category in QUESTIONS:
        parsed = [answer_of(a.get(category)) for a in answers]
        hits = [index for index, (label, _) in enumerate(parsed) if label == "yes"]
        evidence = batch_evidence(model, texts, hits, category, batch_size)
        columns[f"{category}_confidence"] = [score if label == "yes" else np.nan for label, score in parsed]
        columns[f"{category}_evidence"] = [evidence.get(index) for index in range(len(texts))]
    print(f"evidence done in {time.time() - started:.0f}s", flush=True)
    return df.assign(**columns)


def ai_scores(df: pd.DataFrame) -> pd.Series:
    """English only: the detector has no Arabic training data."""
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL).to(DEVICE).eval()
    english = df[df["language"] == "en"]
    scores = []
    with torch.no_grad():
        for start in range(0, len(english), 16):
            batch = tokenizer(english["text"].iloc[start:start + 16].tolist(), padding=True, truncation=True,
                              max_length=AI_MAX_LENGTH, return_tensors="pt").to(DEVICE)
            scores += torch.softmax(model(**batch).logits, dim=1)[:, 1].tolist()
    print(f"AI score computed for {len(english)} English reviews")
    return pd.Series(scores, index=english.index, dtype=float).reindex(df.index)


def burst_flags(df: pd.DataFrame) -> pd.Series:
    """5-star reviews inside a 48-hour window that is unusually busy for that business.

    Only the newest sample has consecutive reviews, so only it is checked.
    """
    flags = pd.Series(False, index=df.index)
    five_star = df[(df["rating"] == 5) & (df["sample"] == "newest")]
    for _, group in five_star.groupby("business"):
        times = group["timestamp_utc"].sort_values()
        days = max((times.iloc[-1] - times.iloc[0]) / pd.Timedelta(days=1), 1)
        usual = len(times) / days * (BURST_WINDOW / pd.Timedelta(days=1))
        needed = max(BURST_MIN_REVIEWS, BURST_FACTOR * usual)
        for index, moment in times.items():
            close = ((times - moment).abs() <= BURST_WINDOW / 2).sum()
            flags[index] = close >= needed
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


def add_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many Google reviews each review we read stands for, using Google's star breakdown.

    The sample over-reads low ratings, and bigger places have more of them to read.
    """
    folder = snapshot_download(DATASET_REPO, repo_type="dataset", allow_patterns="raw/*_newest.json")
    stars = {}
    for path in Path(folder, "raw").glob("*_newest.json"):
        place = json.loads(path.read_text(encoding="utf-8"))
        stars[place["google_id"]] = {int(star): int(count) for star, count in place["reviews_per_score"].items()}
    missing = set(df["google_id"]) - set(stars)
    if missing:
        raise RuntimeError(f"No star breakdown for {missing}")
    read = df.groupby(["google_id", "rating"])["review_id"].transform("size")
    total = pd.Series([stars[g].get(int(r), 0) for g, r in zip(df["google_id"], df["rating"])], index=df.index)
    df["coverage_weight"] = (total / read).clip(lower=1)
    return df


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
    sentences = df["text"].map(lambda text: sum(len(s) >= MIN_SENTENCE_CHARS for s in SENTENCE_BREAK.split(text)))
    for category in QUESTIONS:
        hit = df[f"{category}_confidence"].notna()
        # 1-2 star reviews are never treated as praise, so a real complaint cannot be hidden.
        praise = (df["rating"] >= HIGH_STARS) | ((df["rating"] == MIXED_STARS) & df["positive"])
        df[f"{category}_likely_praise"] = hit & praise
        # The whole review said yes, but no single sentence did.
        unsupported = hit & (df[f"{category}_evidence"] == df["text"]) & (sentences > 1)
        df[f"{category}_unsupported"] = unsupported
        low = (df[f"{category}_confidence"] < LOW_CONFIDENCE) | (df["credibility"] < CREDIBLE) | praise | unsupported
        df[f"{category}_low_confidence"] = hit & low
    return df


def severity_of(hits: pd.DataFrame, category: str, now: pd.Timestamp, place_reviews: int) -> tuple[float, int, int]:
    """Returns (severity, credible complaints, complaints).

    Likely praise and hits with no supporting sentence never add to severity, but stay visible.
    """
    alleging = hits[~hits[f"{category}_likely_praise"]]
    supported = alleging[~alleging[f"{category}_unsupported"]]
    credible = supported[supported["credibility"] >= CREDIBLE]
    if credible.empty and (alleging.empty or category not in NEVER_DROPPED):
        return 0.0, 0, len(alleging)
    used = credible if len(credible) else alleging
    count = credible["coverage_weight"].sum() if len(credible) else LOW_CONFIDENCE_FLOOR_COUNT
    per_thousand = count * 1000 / max(place_reviews, 1)
    ages = (now - used["timestamp_utc"]).dt.days.clip(lower=0)
    recency = (0.5 ** (ages / RECENCY_HALF_LIFE_DAYS)).mean()
    severity = CATEGORY_WEIGHT[category] * used[f"{category}_confidence"].mean() * per_thousand * recency
    return float(severity), len(credible), len(alleging)


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
            severity, credible, count = severity_of(hits, category, now, int(first["place_reviews"]))
            row[f"{category}_severity"] = severity
            row[f"{category}_credible_hits"] = credible
            row[f"{category}_hits"] = count
            row[f"{category}_praise_hits"] = len(hits) - count
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
        hits = group[group[f"{category}_confidence"].notna() & ~group[f"{category}_likely_praise"]]
        hits = hits[~hits[f"{category}_unsupported"]].sort_values(f"{category}_confidence", ascending=False)
        findings.append({
            "category": category,
            "reviews_alleging": int(row[f"{category}_hits"]),
            "credible_reviews": int(row[f"{category}_credible_hits"]),
            "examples": hits[f"{category}_evidence"].head(2).str[:200].tolist(),
        })
    return {"business": row["business"], "type": row["category"],
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-llm-calls", type=int, default=MAX_LLM_CALLS)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--to-smoke", action="store_true", help="push to smoke/, for inspecting a full run")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    classified = classify(load_reviews(args.limit), args.batch_size)
    classified.to_parquet(OUTPUT_DIR / "classified.parquet", index=False)  # Checkpoint for the slowest step.
    reviews = add_credibility(add_coverage(classified))
    summary = add_briefs(summarise(reviews), reviews, args.max_llm_calls)

    reviews.to_parquet(OUTPUT_DIR / "reviews.parquet", index=False)
    summary.to_parquet(OUTPUT_DIR / "businesses.parquet", index=False)
    print(f"saved {OUTPUT_DIR}: {len(reviews)} reviews, {len(summary)} businesses")
    show(summary, reviews)

    if args.no_push:
        print("\nNot pushed (--no-push).")
        return
    folder = "smoke/" if args.limit or args.to_smoke else ""
    hub = HfApi()
    for name in ("reviews.parquet", "businesses.parquet"):
        hub.upload_file(path_or_fileobj=str(OUTPUT_DIR / name), path_in_repo=f"{folder}{name}", repo_id=DATASET_REPO,
                        repo_type="dataset", commit_message=f"Phase 5: {folder}{name}")
    print(f"pushed: https://huggingface.co/datasets/{DATASET_REPO}/tree/main/{folder}")


if __name__ == "__main__":
    main()
