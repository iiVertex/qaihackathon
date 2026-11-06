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
CATEGORY_HELP = {
    "safety": "Danger to people, such as fire risks, injuries or exposed wiring",
    "hygiene": "Cleanliness, such as dirt, pests, bad smells or spoiled food",
    "licensing": "Possible rule breaking, such as alcohol, smoking or permits",
    "pricing_fraud": "Overcharging, hidden fees or refused refunds",
    "service_only": "Rude staff or slow service, with no rule broken",
}
LEVEL_TEXT = {"High": "High concern", "Medium": "Medium concern", "Low": "Low concern", "None": "No complaints"}
STATUSES = ["Confirmed", "False alarm", "Resolved"]
LANGUAGES = {"English": "en", "العربية": "ar"}
METHOD = """
- **Reviews:** up to 50 newest and 50 lowest-rated reviews with text, for each business. Because low ratings are over-sampled, counts are not percentages of all reviews.
- **Concern levels:** each category compares a business with the other 9. High concern is the top three, Medium concern is the next three, and Low concern is the rest. These are comparisons, not safety ratings.
- **Look reliable:** the review passed the credibility checks and has a sentence supporting the complaint.
- **Violations:** a GLiNER2 yes/no question for each category. Results have not been measured against labelled data, so expect false alarms and misses.
- **AI-text check:** English reviews only. On held-out hotel reviews it wrongly flagged about 1 in 7 human reviews, so it only lowers credibility slightly.
- **Credibility and severity:** hand-set weights, not calibrated values. Severity = category weight x confidence x credible complaints per 1,000 Google reviews x recency.
- **Fair ranking:** complaints are scaled using Google's star breakdown, so a business is not ranked on size alone. This is an estimate.
- **Hygiene and safety:** these hits are never dropped, only marked low confidence. Hits on 4 and 5 star reviews, and on positive 3 star reviews, are marked likely praise and do not add to severity. Neither do hits where no single sentence supports the answer.
- **Known weakness:** safety and licensing hits are the noisiest, so check their evidence before acting on them.
"""
MAROON = gr.themes.Color(
    c50="#fdf2f5", c100="#fbe6ec", c200="#f5c6d4", c300="#ec9bb3", c400="#df6a8c", c500="#c8406a",
    c600="#a8234f", c700="#8a1538", c800="#70122f", c900="#5c1029", c950="#360615", name="maroon",
)
THEME = gr.themes.Soft(
    primary_hue=MAROON,
    neutral_hue=gr.themes.colors.zinc,
    spacing_size=gr.themes.sizes.spacing_lg,
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_text_color="*body_text_color",
    block_title_text_color_dark="*body_text_color",
    block_title_text_weight="600",
    block_title_padding="0",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*body_text_color_subdued",
    block_label_text_color_dark="*body_text_color_subdued",
    block_shadow="none",
    block_shadow_dark="none",
    block_padding="18px 20px",
    block_background_fill="*background_fill_secondary",
    block_background_fill_dark="*background_fill_secondary",
    block_border_width="1px",
    block_border_color="*border_color_primary",
    block_border_color_dark="*border_color_primary",
    body_text_color_subdued="*neutral_600",
    body_text_color_subdued_dark="*neutral_400",
    input_shadow="none",
    input_shadow_dark="none",
    input_border_width="1px",
    input_border_color="*neutral_300",
    input_border_color_dark="*neutral_600",
    checkbox_label_shadow="none",
    checkbox_label_shadow_dark="none",
    checkbox_label_border_width="1px",
    checkbox_label_border_color="*neutral_300",
    checkbox_label_border_color_dark="*neutral_600",
    checkbox_border_width="1px",
    checkbox_border_color="*neutral_500",
    checkbox_border_color_dark="*neutral_400",
    layout_gap="22px",
)
# Fonts load here, because Gradio's launch() crashes comparing a custom font list to its built-in themes.
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600'
         '&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap">'
         # Selecting the current name on focus means typing replaces it, so search works straight away.
         "<script>document.addEventListener('focusin',e=>{if(e.target.matches('#ri-search input'))e.target.select()})</script>")
# Spacing scale: 6, 12, 20, 28. Sections sit further apart than the items inside them.
CSS = """
:root,.gradio-container{--font:'IBM Plex Sans','IBM Plex Sans Arabic',system-ui,sans-serif!important}
.gradio-container{max-width:1080px!important;width:100%!important;margin-inline:auto!important}
.html-container{padding:0!important}
.ri-header{display:flex;flex-direction:column;gap:6px}
.ri-header h1{font-size:2rem;line-height:1.2;margin:0;font-weight:600;text-wrap:balance}
.ri-header p{margin:0;color:var(--body-text-color-subdued);font-size:1.02rem;line-height:1.5}
.ri-card{border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);border-radius:12px;padding:24px 26px}
.ri-title{display:flex;flex-direction:column;gap:6px}
.ri-title h2{margin:0;font-size:1.4rem;line-height:1.3;font-weight:600}
.ri-muted{color:var(--body-text-color-subdued);font-size:.88rem;line-height:1.55}
.ri-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-block:20px}
.ri-stats:first-child{margin-top:0}
.ri-stat{display:flex;flex-direction:column;gap:6px;border:1px solid var(--border-color-primary);background:var(--background-fill-primary);border-radius:10px;padding:14px 16px}
.ri-stat b{font-size:1.5rem;line-height:1.2;font-weight:600;font-variant-numeric:tabular-nums}
.ri-subhead{margin:0;font-weight:600;font-size:1.05rem}
.ri-rows{display:flex;flex-direction:column;margin-top:10px}
.ri-row{display:grid;grid-template-columns:minmax(0,1fr) 120px 128px;gap:16px;align-items:center;padding-block:12px;border-top:1px solid var(--border-color-primary)}
.ri-row:first-child{border-top:none}
.ri-row-main,.ri-row-count{display:flex;flex-direction:column;gap:2px;min-width:0}
.ri-row-count{font-variant-numeric:tabular-nums}
.ri-level{justify-self:end;white-space:nowrap;border:1px solid var(--border-color-primary);border-radius:999px;padding:3px 11px;font-size:.8rem;font-weight:600;line-height:1.4}
.ri-card-title{margin:0 0 10px;font-size:1.05rem;font-weight:600}
.ri-check{display:flex!important;flex-direction:row!important;align-items:center;gap:16px;padding-block:10px!important}
.ri-check [data-testid="block-info"]{flex:0 0 200px;margin:0!important;padding:0!important}
.ri-check .wrap{flex:1 1 auto;min-width:0;gap:8px}
.ri-level-high{border-color:rgba(220,38,38,.6);background:rgba(220,38,38,.15)}
.ri-level-medium{border-color:rgba(217,119,6,.65);background:rgba(245,158,11,.16)}
.ri-level-low{border-color:rgba(22,163,74,.55);background:rgba(34,197,94,.14)}
#ri-search input{height:48px;padding-left:44px!important;font-size:1rem;background:var(--input-background-fill) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2371717a' stroke-width='2.2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.8-3.8'/%3E%3C/svg%3E") no-repeat 15px center}
#ri-search .icon-wrap,#ri-search svg.dropdown-arrow{display:none!important}
#ri-search,#ri-search .wrap,#ri-search .wrap-inner{padding:0!important;min-height:0;background:transparent!important;border:none!important;box-shadow:none!important}
#ri-search input{border:1px solid var(--input-border-color);border-radius:10px;width:100%}
.ri-note{margin:20px 0 0}
.ri-stars{position:relative;display:inline-block;white-space:nowrap;line-height:1;letter-spacing:1px;font-size:1rem;font-weight:400;color:var(--neutral-400);font-family:system-ui,sans-serif;vertical-align:baseline}
.ri-stars span{position:absolute;top:0;left:0;bottom:0;overflow:hidden;color:#f5a400}
.ri-tag .ri-stars{font-size:.92rem}
.ri-tags{display:flex;flex-wrap:wrap;gap:6px}
.ri-tag{display:inline-flex;align-items:center;border-radius:6px;padding:3px 9px;font-size:.8rem;line-height:1.35;border:1px solid var(--border-color-primary);background:var(--background-fill-primary)}
.ri-low{border-color:rgba(217,119,6,.7);background:rgba(245,158,11,.14)}
.ri-sample{border-color:rgba(124,58,237,.65);background:rgba(124,58,237,.13)}
.ri-groups{display:flex;flex-direction:column;gap:14px}
details.ri-group{border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);border-radius:12px;padding:0 24px}
details.ri-group summary{cursor:pointer;font-weight:600;font-size:1.04rem;padding:16px 0}
details.ri-group summary:focus-visible{outline:2px solid var(--color-accent);outline-offset:4px;border-radius:6px}
.ri-review{display:flex;flex-direction:column;gap:12px;border-top:1px solid var(--border-color-primary);padding-block:18px}
.ri-review p{margin:0;line-height:1.75;max-width:78ch;overflow-wrap:anywhere}
.ri-review p:dir(rtl){line-height:2;font-size:1.03rem;margin-left:auto}
.ri-review mark{background:rgba(250,204,21,.38);color:inherit;border-radius:3px;padding:2px 3px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.ri-more{padding-block:0 18px}
.ri-brief{margin:0;line-height:1.85;max-width:78ch}
.ri-brief:dir(rtl){line-height:2.05;font-size:1.06rem}
.ri-stack{display:flex;flex-direction:column;gap:12px}
.ri-section-intro{margin:0 0 4px;color:var(--body-text-color-subdued);line-height:1.6}
.ri-table-wrap{overflow-x:auto}
table.ri-table{width:100%;border-collapse:collapse;font-size:.95rem;font-variant-numeric:tabular-nums}
table.ri-table th,table.ri-table td{padding:12px 14px;border-bottom:1px solid var(--border-color-primary);text-align:start;white-space:nowrap}
table.ri-table td:nth-child(2){white-space:normal;min-width:180px}
.tabitem{padding:12px 0 0!important;border:none!important;background:transparent!important;min-width:0;max-width:100%}
@media (max-width:620px){
  .ri-header{padding-top:46px}
  .ri-card{padding:18px}
  .ri-stat:first-child{grid-column:1/-1}
  .ri-row{grid-template-columns:minmax(0,1fr) auto;row-gap:8px}
  .ri-row-main{grid-column:1/-1}
  .ri-check{flex-direction:column!important;align-items:flex-start;gap:8px}
  .ri-check [data-testid="block-info"]{flex:0 0 auto}
  details.ri-group{padding:0 18px}
  .ri-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  table.ri-table th,table.ri-table td{white-space:normal;padding:10px 8px}
  table.ri-table td:nth-child(2){min-width:0}
  table.ri-table th:nth-child(3),table.ri-table td:nth-child(3),
  table.ri-table th:nth-child(6),table.ri-table td:nth-child(6){display:none}
}
"""


def data_path(name: str) -> str:
    if LOCAL_DATA_DIR:
        return str(Path(LOCAL_DATA_DIR) / name)
    return hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


BUSINESSES = pd.read_parquet(data_path("businesses.parquet"))
REVIEWS = pd.read_parquet(data_path("reviews.parquet"))
SAMPLES = {r["business"]: r for r in json.loads(Path(data_path("mock_inspections.json")).read_text(encoding="utf-8"))["records"]}

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


def stars_html(value: float) -> str:
    """Google-style stars. The gold layer is cut to size, so 3.9 fills 78% of the row."""
    label = f"{value:g} out of 5 stars"
    return (f"<span class='ri-stars' role='img' aria-label='{label}' title='{label}'>"
            f"<span style='width:{value / 5 * 100:.0f}%'>★★★★★</span>★★★★★</span>")


def business_row(name: str) -> pd.Series:
    return BUSINESSES[BUSINESSES["business"] == name].iloc[0]


def picker_choices() -> list[tuple[str, str]]:
    return [(f"{r['rank']}. {r['business']}", r["business"]) for _, r in BUSINESSES.iterrows()]


def ranking_html() -> str:
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{esc(r['business'])}</td><td>{esc(r['category'])}</td><td>{r['risk_score']:.2f}</td>"
        f"<td>{r['place_rating']} {stars_html(r['place_rating'])}</td><td>{r['reviews_scored']}</td></tr>"
        for _, r in BUSINESSES.iterrows()
    )
    return ("<div class='ri-card ri-table-wrap'><table class='ri-table'><thead><tr><th>Rank</th><th>Business</th>"
            "<th>Type</th><th>Risk</th><th>Google rating</th><th>Reviews analysed</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def level_of(row: pd.Series, category: str) -> str:
    """Compares a business with the others in one category. The levels are relative, not absolute."""
    if row[f"{category}_hits"] == 0:
        return "None"
    position = int((BUSINESSES[f"{category}_severity"] > row[f"{category}_severity"]).sum()) + 1
    return "High" if position <= 3 else "Medium" if position <= 6 else "Low"


def category_row(row: pd.Series, category: str) -> str:
    hits, credible = row[f"{category}_hits"], row[f"{category}_credible_hits"]
    level = level_of(row, category)
    count = f"{hits} {'review' if hits == 1 else 'reviews'}" if hits else "0 reviews"
    reliable = f"{credible} look reliable" if hits else "Nothing to check"
    return (f"<div class='ri-row'><div class='ri-row-main'><strong>{CATEGORY_LABELS[category]}</strong>"
            f"<span class='ri-muted'>{CATEGORY_HELP[category]}</span></div>"
            f"<div class='ri-row-count'><strong>{count}</strong><span class='ri-muted'>{reliable}</span></div>"
            f"<span class='ri-level ri-level-{level.lower()}'>{LEVEL_TEXT[level]}</span></div>")


def risk_card(row: pd.Series) -> str:
    rows = "".join(category_row(row, category) for category in CATEGORY_ORDER)
    stats = (
        ("Risk rank", f"{row['rank']} of {len(BUSINESSES)}"),
        ("Risk score", f"{row['risk_score']:.2f}"),
        ("Google rating", f"{row['place_rating']} {stars_html(row['place_rating'])}"),
        ("Reviews analysed", row["reviews_scored"]),
        ("Average credibility", f"{row['mean_credibility']:.2f}"),
    )
    tiles = "".join(f"<div class='ri-stat'><span class='ri-muted'>{label}</span><b>{value}</b></div>"
                    for label, value in stats)
    return f"""<div class='ri-card'>
      <div class='ri-title'><h2>{esc(row['business'])}</h2>
      <span class='ri-muted'>{esc(row['category']).capitalize()} with {row['place_reviews']} Google reviews</span></div>
      <div class='ri-stats'>{tiles}</div>
      <div class='ri-title'><p class='ri-subhead'>What reviewers are complaining about</p>
      <span class='ri-muted'>Concern level compares this business with the other {len(BUSINESSES) - 1}. Read the reviews in the Complaints tab below.</span></div>
      <div class='ri-rows'>{rows}</div>
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
        f"<span class='ri-tag'>{stars_html(review['rating'])}</span>",
        f"<span class='ri-tag'>{review['timestamp_utc']:%b %Y}</span>",
        f"<span class='ri-tag'>confidence {review[f'{category}_confidence']:.2f}</span>",
        f"<span class='ri-tag'>credibility {review['credibility']:.2f}</span>",
        "<span class='ri-tag ri-low'>likely praise</span>" if praise else "",
        "<span class='ri-tag ri-low'>low confidence</span>" if low and not praise else "",
        whole,
    ]
    return (f"<div class='ri-review'><div class='ri-tags'>{''.join(tags)}</div>"
            f"<p dir='auto'>{highlight(review['text'], evidence)}</p></div>")


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
        more = f"<div class='ri-muted ri-more'>{hidden} more not shown.</div>" if hidden > 0 else ""
        sections += (f"<details class='ri-group'><summary>{CATEGORY_LABELS[category]} ({len(hits) - praise}{extra})</summary>"
                     f"{body}{more}</details>")
    if not sections:
        return "<div class='ri-card'>No complaints found in the reviews analysed.</div>"
    return f"<div class='ri-groups'>{sections}</div>"


def brief_html(name: str, language: str) -> str:
    code = LANGUAGES[language]
    row = business_row(name)
    source = str(row[f"brief_source_{code}"])
    note = "Written by an AI model from the data above." if source.startswith("llm") else "Written from a fixed template."
    direction = "rtl" if code == "ar" else "ltr"
    title = report.TEXT[code]["brief"]
    return (f"<div class='ri-stack'><div class='ri-card' dir='{direction}' lang='{code}'>"
            f"<h3 class='ri-card-title'>{title}</h3><p class='ri-brief'>"
            f"{esc(row[f'brief_{code}'])}</p></div><span class='ri-muted'>{note}</span></div>")


def sample_html(name: str) -> str:
    sample = SAMPLES.get(name)
    if not sample:
        return ""
    return (f"<div class='ri-card ri-stack'><div class='ri-tags'><span class='ri-tag ri-sample'>sample data</span></div>"
            f"<p style='margin:0;line-height:1.7'>Previous inspection: {esc(sample['last_visit'])}, "
            f"{esc(sample['visit_type'])}. {esc(sample['outcome'])}.</p></div>")


def show_business(name: str, language: str):
    row = business_row(name)
    radios = [gr.update(label=f"{CATEGORY_LABELS[c]} ({row[f'{c}_hits']} alleged)", value=None) for c in CATEGORY_ORDER]
    return [risk_card(row), complaints_html(name), brief_html(name, language), sample_html(name),
            *radios, "", "", gr.update(value=None, visible=False)]


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


def make_pdf(name: str, language: str, notes: str, *values):
    row = business_row(name).to_dict()
    reviews = REVIEWS[REVIEWS["business"] == name].to_dict("records")
    try:
        path = report.build_report(row, reviews, len(BUSINESSES), LANGUAGES[language], collect_checklist(values),
                                   notes, SAMPLES.get(name))
        return gr.update(value=path, visible=True)
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
        rows += (f"<div class='ri-review'><strong>{CATEGORY_LABELS[category]}</strong>"
                 f"<div class='ri-tags'>{tags}</div><p dir='auto'>{highlight(text, evidence)}</p></div>")
    ai_text = f"AI-text score {ai_score:.2f}" if ai_score is not None else "AI-text check not available for this language"
    timing = f"Models loaded in {loaded - started:.1f}s, analysis took {time.time() - loaded:.1f}s."
    return f"""<div class='ri-card'>
      <div class='ri-stats'>
        <div class='ri-stat'><span class='ri-muted'>Credibility estimate</span><b>{credibility:.2f}</b></div>
        <div class='ri-stat'><span class='ri-muted'>Review tone</span><b>{'Positive' if positive else 'Not positive'}</b></div>
      </div>
      {rows or 'No violation categories matched this review.'}
      <p class='ri-muted ri-note'>{ai_text}. Reviewer history and bursts need other reviews,
      so they are not part of this estimate. {timing}</p></div>"""


def intro(text: str) -> str:
    return f"<p class='ri-section-intro'>{text}</p>"


with gr.Blocks(title="Review Inspector", fill_width=True) as demo:
    gr.HTML("<div class='ri-header'><h1>Review Inspector</h1>"
            "<p>Ranked evidence from public reviews, for inspectors in Qatar.</p></div>")

    first = BUSINESSES.iloc[0]["business"]
    picker = gr.Dropdown(picker_choices(), value=first, label="Search for a business", show_label=False,
                         container=False, filterable=True, elem_id="ri-search")
    risk = gr.HTML(risk_card(business_row(first)))

    with gr.Tabs():
        with gr.Tab("Complaints"):
            gr.HTML(intro("Safety comes first. Highlighted text is the sentence that supports the complaint."))
            complaints = gr.HTML(complaints_html(first))

        with gr.Tab("Brief and report"):
            language = gr.Radio(list(LANGUAGES), value="English", label="Report language")
            brief = gr.HTML(brief_html(first, "English"))
            sample = gr.HTML(sample_html(first))
            export = gr.Button("Create PDF report", variant="primary")
            pdf = gr.File(label="PDF report", interactive=False, visible=False)

        with gr.Tab("Checklist"):
            gr.HTML(intro("Record what you found on site. Your choices also appear in the PDF report."))
            radios = [gr.Radio(STATUSES, label=f"{CATEGORY_LABELS[c]} ({first_hits} alleged)", elem_classes="ri-check")
                      for c, first_hits in ((c, business_row(first)[f"{c}_hits"]) for c in CATEGORY_ORDER)]
            notes = gr.Textbox(label="Notes", lines=3)
            save = gr.Button("Save checklist", variant="primary")
            saved = gr.Markdown()

        with gr.Tab("Try a review"):
            gr.HTML(intro("Paste a review to run the checks live. The first run loads the models and takes longer."))
            with gr.Row():
                pasted = gr.Textbox(label="Review text", lines=5, scale=4)
                stars = gr.Dropdown([("Not given", "Not given")] + [("★" * n + "☆" * (5 - n), str(n)) for n in range(1, 6)],
                                    value="Not given", label="Star rating", scale=1)
            run = gr.Button("Analyse review", variant="primary")
            result = gr.HTML()

        with gr.Tab("All businesses"):
            gr.HTML(ranking_html())

        with gr.Tab("How this works"):
            gr.Markdown(METHOD)

    picker.change(show_business, [picker, language], [risk, complaints, brief, sample, *radios, notes, saved, pdf],
                  api_name="show_business")
    language.change(brief_html, [picker, language], brief, api_name="brief")
    save.click(save_checklist, [picker, notes, *radios], saved, api_name="save_checklist")
    export.click(make_pdf, [picker, language, notes, *radios], pdf, api_name="make_pdf")
    run.click(analyse, [pasted, stars], result, api_name="analyse")

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=FONTS, ssr_mode=False)
