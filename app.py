"""Review Inspector: reads precomputed results. Models load only for the live box."""
import html
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import CommitScheduler, hf_hub_download

import report

DATASET_REPO = "Hermesione/reviewere-dataset"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR")
CHECKLIST_DIR = Path("checklist_log")
SCHEDULER_MINUTES = 2
MAX_SHOWN = 6
MAX_LIVE_CHARS = 2000
CREDIBLE = 0.5
LOW_CONFIDENCE = 0.6

CATEGORY_ORDER = report.CATEGORY_ORDER
CATEGORY_LABELS = report.TEXT["en"]["categories"]
STATUSES = ["Confirmed", "False alarm", "Resolved"]
LANGUAGES = {"English": "en", "العربية": "ar"}
BANNER = (
    "<strong>Unverified allegations from public Google Maps reviews.</strong> These are not inspection findings. "
    "Credibility is an estimate, never a verdict that a review is fake."
)
SAMPLE_NOTE = (
    "This sample deliberately includes each business's lowest-rated reviews, "
    "so counts are not percentages of all reviews."
)
METHOD = """
- **Reviews:** up to 50 newest and 50 lowest-rated reviews with text, for each business.
- **Violations:** a GLiNER2 yes/no question for each category. Results have not been measured against labelled data, so expect false alarms and misses.
- **AI-text check:** English reviews only. On held-out hotel reviews it wrongly flagged about 1 in 7 human reviews, so it only lowers credibility slightly.
- **Credibility and severity:** hand-set weights, not calibrated values. Severity = category weight x confidence x credible complaints per 1,000 Google reviews x recency.
- **Fair ranking:** complaints are scaled using Google's star breakdown, so a business is not ranked on size alone. This is an estimate.
- **Hygiene and safety:** these hits are never dropped, only marked low confidence. Hits on 4 and 5 star reviews, and on positive 3 star reviews, are marked likely praise and do not add to severity. Neither do hits where no single sentence supports the answer.
- **Known weakness:** safety and licensing hits are the noisiest, so check their evidence before acting on them.
"""
CSS = """
.ri-banner{border:1px solid rgba(217,119,6,.55);background:rgba(245,158,11,.14);border-radius:8px;padding:10px 14px;line-height:1.5}
.ri-card{border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);border-radius:10px;padding:14px 16px;margin-bottom:10px}
.ri-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:10px 0}
.ri-stat b{display:block;font-size:1.45rem;line-height:1.2}
.ri-muted{opacity:.75;font-size:.87rem}
.ri-bar-row{display:grid;grid-template-columns:110px 1fr 90px;gap:10px;align-items:center;margin:5px 0;font-size:.9rem}
.ri-bar{background:var(--border-color-primary);border-radius:4px;height:9px;overflow:hidden}
.ri-bar span{display:block;height:100%;background:var(--color-accent)}
.ri-tag{display:inline-block;border-radius:5px;padding:0 7px;font-size:.78rem;border:1px solid var(--border-color-primary);margin-inline-end:4px}
.ri-low{border-color:rgba(217,119,6,.7);background:rgba(245,158,11,.14)}
.ri-sample{border-color:rgba(124,58,237,.65);background:rgba(124,58,237,.13)}
.ri-review{border-top:1px solid var(--border-color-primary);padding:9px 0}
.ri-review p{margin:6px 0 0;line-height:1.6}
.ri-review mark{background:rgba(250,204,21,.38);color:inherit;border-radius:3px;padding:0 2px}
details.ri-group{margin-bottom:10px}
details.ri-group summary{cursor:pointer;font-weight:600;padding:4px 0}
table.ri-table{width:100%;border-collapse:collapse;font-size:.92rem}
table.ri-table th,table.ri-table td{padding:6px 8px;border-bottom:1px solid var(--border-color-primary);text-align:start}
"""


def data_path(name: str) -> str:
    if LOCAL_DATA_DIR:
        return str(Path(LOCAL_DATA_DIR) / name)
    return hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


BUSINESSES = pd.read_parquet(data_path("businesses.parquet"))
REVIEWS = pd.read_parquet(data_path("reviews.parquet"))
SAMPLES = {r["business"]: r for r in json.loads(Path(data_path("mock_inspections.json")).read_text(encoding="utf-8"))["records"]}
MAX_SEVERITY = max(BUSINESSES[[f"{c}_severity" for c in CATEGORY_ORDER]].max().max(), 1e-9)

CHECKLIST_DIR.mkdir(exist_ok=True)
CHECKLIST_FILE = CHECKLIST_DIR / f"{uuid.uuid4().hex}.jsonl"
# Local runs keep the checklist on disk, so testing never writes to the dataset repo.
SCHEDULER = (
    CommitScheduler(repo_id=DATASET_REPO, repo_type="dataset", folder_path=CHECKLIST_DIR, path_in_repo="checklist_log",
                    every=SCHEDULER_MINUTES, token=os.environ.get("HF_TOKEN"))
    if os.environ.get("SPACE_ID") else None
)
CHECKLIST_LOCK = SCHEDULER.lock if SCHEDULER else threading.Lock()

LIVE = {}
LIVE_LOCK = threading.Lock()


def esc(value) -> str:
    """Review text is untrusted, so it is always escaped before going into HTML."""
    return html.escape(str(value))


def business_row(name: str) -> pd.Series:
    return BUSINESSES[BUSINESSES["business"] == name].iloc[0]


def picker_choices() -> list[tuple[str, str]]:
    return [(f"{r['rank']}. {r['business']} ({r['category']}), risk {r['risk_score']:.2f}", r["business"])
            for _, r in BUSINESSES.iterrows()]


def ranking_html() -> str:
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{esc(r['business'])}</td><td>{esc(r['category'])}</td><td>{r['risk_score']:.2f}</td>"
        f"<td>{r['place_rating']}</td><td>{r['reviews_scored']}</td></tr>"
        for _, r in BUSINESSES.iterrows()
    )
    return ("<table class='ri-table'><thead><tr><th>Rank</th><th>Business</th><th>Type</th><th>Risk</th>"
            f"<th>Google rating</th><th>Reviews analysed</th></tr></thead><tbody>{rows}</tbody></table>")


def risk_card(row: pd.Series) -> str:
    bars = ""
    for category in CATEGORY_ORDER:
        share = 100 * row[f"{category}_severity"] / MAX_SEVERITY
        bars += (f"<div class='ri-bar-row'><span>{CATEGORY_LABELS[category]}</span>"
                 f"<div class='ri-bar'><span style='width:{share:.0f}%'></span></div>"
                 f"<span class='ri-muted'>{row[f'{category}_hits']} alleged</span></div>")
    english = row["ai_check_available"]
    return f"""<div class='ri-card'>
      <h3 style='margin:0'>{esc(row['business'])}</h3>
      <div class='ri-muted'>{esc(row['category'])}, {row['place_reviews']} Google reviews</div>
      <div class='ri-stats'>
        <div class='ri-stat'><b>{row['rank']} of {len(BUSINESSES)}</b><span class='ri-muted'>Risk rank</span></div>
        <div class='ri-stat'><b>{row['risk_score']:.2f}</b><span class='ri-muted'>Risk score</span></div>
        <div class='ri-stat'><b>{row['place_rating']}</b><span class='ri-muted'>Google rating</span></div>
        <div class='ri-stat'><b>{row['reviews_scored']}</b><span class='ri-muted'>Reviews analysed</span></div>
        <div class='ri-stat'><b>{row['mean_credibility']:.2f}</b><span class='ri-muted'>Average credibility</span></div>
      </div>
      {bars}
      <div class='ri-muted' style='margin-top:8px'>AI-text check covered {english} English reviews. {SAMPLE_NOTE}</div>
    </div>"""


def highlight(text: str, evidence: str) -> str:
    if evidence and evidence != text and evidence in text:
        before, after = text.split(evidence, 1)
        return f"{esc(before)}<mark>{esc(evidence)}</mark>{esc(after)}"
    return esc(text)


def review_html(review: pd.Series, category: str) -> str:
    low = review[f"{category}_low_confidence"]
    praise = review[f"{category}_likely_praise"]
    evidence = review[f"{category}_evidence"]
    whole = "" if evidence != review["text"] else "<span class='ri-tag'>whole review</span>"
    tags = [
        f"<span class='ri-tag'>{review['rating']}/5</span>",
        f"<span class='ri-tag'>{review['timestamp_utc']:%b %Y}</span>",
        f"<span class='ri-tag'>confidence {review[f'{category}_confidence']:.2f}</span>",
        f"<span class='ri-tag'>credibility {review['credibility']:.2f}</span>",
        "<span class='ri-tag ri-low'>likely praise</span>" if praise else "",
        "<span class='ri-tag ri-low'>low confidence</span>" if low and not praise else "",
        whole,
    ]
    return f"<div class='ri-review'>{''.join(tags)}<p dir='auto'>{highlight(review['text'], evidence)}</p></div>"


def complaints_html(name: str) -> str:
    group = REVIEWS[REVIEWS["business"] == name]
    sections = ""
    for index, category in enumerate(CATEGORY_ORDER):
        hits = group[group[f"{category}_confidence"].notna()]
        if hits.empty:
            continue
        order = [f"{category}_likely_praise", f"{category}_low_confidence", f"{category}_confidence"]
        ranked = hits.sort_values(order, ascending=[True, True, False])
        praise = int(hits[f"{category}_likely_praise"].sum())
        extra = f", plus {praise} likely praise" if praise else ""
        body = "".join(review_html(review, category) for _, review in ranked.head(MAX_SHOWN).iterrows())
        hidden = len(hits) - MAX_SHOWN
        more = f"<div class='ri-muted'>{hidden} more not shown.</div>" if hidden > 0 else ""
        opened = " open" if category in ("safety", "hygiene") else ""
        sections += (f"<details class='ri-group'{opened}><summary>{CATEGORY_LABELS[category]} ({len(hits) - praise}{extra})</summary>"
                     f"{body}{more}</details>")
    return f"<div class='ri-card'>{sections or 'No complaints found in the reviews analysed.'}</div>"


def brief_html(name: str, language: str) -> str:
    code = LANGUAGES[language]
    row = business_row(name)
    source = str(row[f"brief_source_{code}"])
    note = "Written by an AI model from the data above." if source.startswith("llm") else "Written from a fixed template."
    direction = "rtl" if code == "ar" else "ltr"
    return (f"<div class='ri-card' dir='{direction}' lang='{code}'><p style='margin:0;line-height:1.7'>"
            f"{esc(row[f'brief_{code}'])}</p></div><div class='ri-muted'>{note}</div>")


def sample_html(name: str) -> str:
    sample = SAMPLES.get(name)
    if not sample:
        return ""
    return (f"<div class='ri-card'><span class='ri-tag ri-sample'>sample data</span>"
            f"<p style='margin:8px 0 0'>Previous inspection: {esc(sample['last_visit'])}, {esc(sample['visit_type'])}. "
            f"{esc(sample['outcome'])}.</p></div>")


def show_business(name: str, language: str):
    row = business_row(name)
    radios = [gr.update(label=f"{CATEGORY_LABELS[c]} ({row[f'{c}_hits']} alleged)", value=None) for c in CATEGORY_ORDER]
    return [risk_card(row), complaints_html(name), brief_html(name, language), sample_html(name),
            *radios, "", "", None]


def collect_checklist(values) -> dict:
    return {category: value for category, value in zip(CATEGORY_ORDER, values) if value}


def save_checklist(name: str, notes: str, *values) -> str:
    checklist = collect_checklist(values)
    if not checklist and not notes.strip():
        raise gr.Error("Choose at least one status or add a note first.")
    entry = {"time": datetime.now(timezone.utc).isoformat(), "business": name, "checklist": checklist, "notes": notes}
    with CHECKLIST_LOCK:
        with CHECKLIST_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    where = f"The dataset repo updates within {SCHEDULER_MINUTES} minutes." if SCHEDULER else "Saved on this machine only."
    return f"Saved {len(checklist)} statuses for {name}. {where}"


def make_pdf(name: str, language: str, notes: str, *values) -> str:
    row = business_row(name).to_dict()
    reviews = REVIEWS[REVIEWS["business"] == name].to_dict("records")
    try:
        return report.build_report(row, reviews, len(BUSINESSES), LANGUAGES[language], collect_checklist(values),
                                   notes, SAMPLES.get(name))
    except FileNotFoundError as error:
        raise gr.Error(f"PDF font is missing: {error}") from error


def live_models() -> dict:
    with LIVE_LOCK:
        if not LIVE:
            import torch
            from lingua import Language, LanguageDetectorBuilder
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            import pipeline

            token = os.environ.get("HF_TOKEN")
            LIVE["pipeline"] = pipeline
            LIVE["torch"] = torch
            LIVE["gliner"] = pipeline.AutoExtractor.from_pretrained(pipeline.GLINER_MODEL)
            LIVE["tokenizer"] = AutoTokenizer.from_pretrained(pipeline.DETECTOR_MODEL, token=token)
            LIVE["detector"] = AutoModelForSequenceClassification.from_pretrained(
                pipeline.DETECTOR_MODEL, token=token).eval()
            LIVE["language"] = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.ARABIC, Language.FRENCH, Language.SPANISH, Language.TURKISH,
                Language.RUSSIAN, Language.HINDI, Language.URDU).build()
    return LIVE


def analyse(text: str, stars: str, progress=gr.Progress()) -> str:
    text = " ".join((text or "").split())[:MAX_LIVE_CHARS]
    if not text:
        raise gr.Error("Paste a review first.")
    started = time.time()
    if not LIVE:
        progress(0.1, desc="Loading models, first run only")
    models = live_models()
    loaded = time.time()
    pipeline, torch = models["pipeline"], models["torch"]
    progress(0.4, desc="Checking for violations")
    raw = models["gliner"].classify_text(text, pipeline.TASKS, include_confidence=True)
    positive = pipeline.answer_of(raw.get("positive"))[0] == "yes"

    detected = models["language"].detect_language_of(text)
    english = detected is not None and detected.iso_code_639_1.name == "EN"
    ai_score = None
    if english:
        batch = models["tokenizer"](text, truncation=True, max_length=pipeline.AI_MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            ai_score = torch.softmax(models["detector"](**batch).logits, dim=1)[0, 1].item()

    rating = int(stars) if stars.isdigit() else None
    mismatch = rating is not None and ((rating >= 4 and not positive) or (rating <= 2 and positive))
    penalty = (pipeline.PENALTY_RATING_MISMATCH if mismatch else 0) + (ai_score or 0) * pipeline.PENALTY_AI_MAX
    credibility = max(1 - penalty, pipeline.CREDIBILITY_FLOOR)

    progress(0.7, desc="Finding evidence")
    rows = ""
    for category in CATEGORY_ORDER:
        label, confidence = pipeline.answer_of(raw.get(category))
        if label != "yes":
            continue
        evidence = pipeline.best_evidence(models["gliner"], text, category)
        praise = rating is not None and (rating >= pipeline.HIGH_STARS or (rating == pipeline.MIXED_STARS and positive))
        low = confidence < LOW_CONFIDENCE or credibility < CREDIBLE or praise
        severity = pipeline.CATEGORY_WEIGHT[category] * confidence
        tags = (f"<span class='ri-tag'>confidence {confidence:.2f}</span>"
                f"<span class='ri-tag'>severity {severity:.2f}</span>"
                + ("<span class='ri-tag ri-low'>low confidence</span>" if low else ""))
        rows += (f"<div class='ri-review'><strong>{CATEGORY_LABELS[category]}</strong> {tags}"
                 f"<p dir='auto'>{highlight(text, evidence)}</p></div>")
    ai_text = f"AI-text score {ai_score:.2f}" if ai_score is not None else "AI-text check not available for this language"
    timing = f"Models loaded in {loaded - started:.1f}s, analysis took {time.time() - loaded:.1f}s."
    return f"""<div class='ri-card'>
      <div class='ri-stats'>
        <div class='ri-stat'><b>{credibility:.2f}</b><span class='ri-muted'>Credibility estimate</span></div>
        <div class='ri-stat'><b>{'Positive' if positive else 'Not positive'}</b><span class='ri-muted'>Review tone</span></div>
      </div>
      {rows or 'No violation categories matched this review.'}
      <div class='ri-muted' style='margin-top:8px'>{ai_text}. Reviewer history and bursts need other reviews,
      so they are not part of this estimate. {timing}</div></div>"""


with gr.Blocks(title="Review Inspector") as demo:
    gr.Markdown("# Review Inspector\nRanked review evidence for inspectors in Qatar.")
    gr.HTML(f"<div class='ri-banner'>{BANNER}</div>")
    with gr.Accordion("All businesses ranked", open=False):
        gr.HTML(ranking_html())

    first = BUSINESSES.iloc[0]["business"]
    picker = gr.Dropdown(picker_choices(), value=first, label="Business")
    risk = gr.HTML(risk_card(business_row(first)))

    gr.Markdown("## Complaints\nSafety comes first. Highlighted text is the evidence sentence.")
    complaints = gr.HTML(complaints_html(first))

    gr.Markdown("## Brief")
    language = gr.Radio(list(LANGUAGES), value="English", label="Report language")
    brief = gr.HTML(brief_html(first, "English"))
    sample = gr.HTML(sample_html(first))

    gr.Markdown("## Inspector checklist")
    radios = [gr.Radio(STATUSES, label=f"{CATEGORY_LABELS[c]} ({first_hits} alleged)")
              for c, first_hits in ((c, business_row(first)[f"{c}_hits"]) for c in CATEGORY_ORDER)]
    notes = gr.Textbox(label="Notes", lines=2)
    with gr.Row():
        save = gr.Button("Save checklist", variant="primary")
        export = gr.Button("Create PDF report")
    saved = gr.Markdown()
    pdf = gr.File(label="PDF report", interactive=False)

    gr.Markdown("## Try a review\nPaste a review to run the checks live. The first run loads the models and takes longer.")
    with gr.Row():
        pasted = gr.Textbox(label="Review text", lines=4, scale=4)
        stars = gr.Dropdown(["Not given", "1", "2", "3", "4", "5"], value="Not given", label="Star rating", scale=1)
    run = gr.Button("Analyse review")
    result = gr.HTML()

    with gr.Accordion("How this works", open=False):
        gr.Markdown(METHOD)

    picker.change(show_business, [picker, language], [risk, complaints, brief, sample, *radios, notes, saved, pdf],
                  api_name="show_business")
    language.change(brief_html, [picker, language], brief, api_name="brief")
    save.click(save_checklist, [picker, notes, *radios], saved, api_name="save_checklist")
    export.click(make_pdf, [picker, language, notes, *radios], pdf, api_name="make_pdf")
    run.click(analyse, [pasted, stars], result, api_name="analyse")

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(), ssr_mode=False)
