"""Review Inspector: reads precomputed results. Models load only for the live box."""
import html
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download

import accounts
import report

DATASET_REPO = "Hermesione/reviewere-dataset"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR")
MAX_SHOWN = 6
MIN_SENTENCE_MATCH = 0.6
MAX_NOTES = 5000

CATEGORY_ORDER = report.CATEGORY_ORDER
CATEGORY_LABELS = report.TEXT["en"]["categories"]
CATEGORY_HELP = {
    "safety": "Danger to people, such as fire hazards, injuries or exposed wiring",
    "hygiene": "Cleanliness, such as dirt, pests, bad smells or spoiled food",
    "licensing": "Possible rule breaking, such as alcohol, smoking or permits",
    "pricing_fraud": "Overcharging, hidden fees or refused refunds",
    "service_only": "Rude staff or slow service, with no rule broken",
}
LEVEL_TEXT = {"High": "High concern", "Medium": "Medium concern", "Low": "Low concern", "None": "No complaints"}
STATUSES = ["Confirmed", "Ongoing", "False alarm", "Resolved"]
LANGUAGES = {"English": "en", "العربية": "ar"}
SIGN_IN_LABEL = "Sign in or create account"
SIGNED_OUT = {}  # Browser storage keeps its old value if given None, so sign-out must store something empty.
SCREEN_SIZE = 16  # Items refresh() returns: 4 sections, 5 statuses, notes, saved, pdf, mode, history, account, session.
METHOD = """
- **Reviews:** up to 50 newest and 50 lowest-rated reviews with text, for each business. Because low ratings are over-sampled, counts are not percentages of all reviews.
- **Priority score:** adds up the severity of each category. A higher score means look at this business first. It compares these businesses and is not a safety rating.
- **Concern levels:** each category compares a business with the other 9. High concern is the top three, Medium concern is the next three, and Low concern is the rest. These are comparisons, not safety ratings.
- **Look reliable:** the review passed the credibility checks and has a sentence supporting the complaint.
- **Violations:** a GLiNER2 yes/no question for each category. Results have not been measured against labelled data, so expect false alarms and misses.
- **AI-text check:** English reviews only. On held-out hotel reviews it wrongly flagged about 1 in 7 human reviews, so it only lowers credibility slightly.
- **Credibility and severity:** hand-set weights, not calibrated values. Severity = category weight x confidence x credible complaints per 1,000 Google reviews x recency.
- **Fair ranking:** complaints are scaled using Google's star breakdown, so a business is not ranked on size alone. This is an estimate.
- **Hygiene and safety:** these hits are never dropped, only marked low confidence. Hits on 4 and 5 star reviews, and on positive 3 star reviews, are marked likely praise and do not add to severity. Neither do hits where no single sentence supports the answer.
- **Translations:** reviews in other languages are shown in English, translated by an AI model (GLM-5.3), with the original language in brackets. Very short reviews are left as written.
- **Demo and accounts:** anyone can try the app. Only inspectors with an account save their work, and each inspector sees only their own.
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
.ri-review p.ri-translated-from{color:var(--body-text-color-subdued);font-size:.88rem;line-height:1.5;margin-top:-6px}
.ri-review mark{background:rgba(250,204,21,.38);color:inherit;border-radius:3px;padding:2px 3px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.ri-more{padding-block:0 18px}
.ri-mode{border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);border-radius:10px;padding:10px 16px;line-height:1.55}
.ri-mode-on{border-color:rgba(22,163,74,.55);background:rgba(34,197,94,.12)}
#ri-top{align-items:center;gap:14px}
#ri-login{height:48px;white-space:nowrap}
.ri-auth-actions{gap:10px}
.ri-hist-list{display:flex;flex-direction:column;gap:12px}
.ri-hist{display:flex;flex-direction:column;gap:10px}
.ri-hist-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 16px}
.ri-hist-head strong{font-size:1.05rem}
.ri-hist p{margin:0;line-height:1.65;overflow-wrap:anywhere}
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
TRANSLATIONS = {
    row.text: row.text_en for row in pd.read_parquet(data_path("translations.parquet")).itertuples()
    if isinstance(row.text_en, str) and row.text_en.strip().lower() != row.text.strip().lower()
}
SAMPLES = {r["business"]: r for r in json.loads(Path(data_path("mock_inspections.json")).read_text(encoding="utf-8"))["records"]}


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
            "<th>Type</th><th>Priority</th><th>Google rating</th><th>Reviews analysed</th></tr></thead>"
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
        ("Priority rank", f"{row['rank']} of {len(BUSINESSES)}"),
        ("Priority score", f"{row['risk_score']:.2f}"),
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


def highlight_translation(english: str, evidence: str | None) -> str:
    """The evidence sentence was translated on its own, so find the closest sentence in the full translation."""
    if not evidence or evidence == english:
        return esc(english)
    sentences = re.split(r"(?<=[.!?])\s+", english)
    best = max(sentences, key=lambda s: SequenceMatcher(None, s.lower(), evidence.lower()).ratio())
    if SequenceMatcher(None, best.lower(), evidence.lower()).ratio() < MIN_SENTENCE_MATCH:
        return esc(english)
    return highlight(english, best)


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
    english = TRANSLATIONS.get(review["text"])
    if english:
        body = (f"<p dir='ltr' lang='en'>{highlight_translation(english, TRANSLATIONS.get(evidence))}</p>"
                f"<p class='ri-translated-from'>({esc(report.translated_from(review['text'], review['language']))})</p>")
    else:
        body = f"<p dir='auto'>{highlight(review['text'], evidence)}</p>"
    return f"<div class='ri-review'><div class='ri-tags'>{''.join(tags)}</div>{body}</div>"


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


def saved_date(entry: dict) -> str:
    return datetime.fromisoformat(entry["time"]).strftime("%d %b %Y")


def latest_entry(name: str, entries: list[dict]) -> dict | None:
    return next((entry for entry in reversed(entries) if entry["business"] == name), None)


def last_saved_text(entry: dict | None, signed_in: bool) -> str:
    if not entry:
        return ""
    if not signed_in:
        return "Demo only: this is kept until you leave the page."
    return f"Last saved on {saved_date(entry)}. Your earlier answers are filled in below."


def mode_html(email: str | None) -> str:
    if email:
        return (f"<div class='ri-mode ri-mode-on'><strong>Signed in as {esc(email)}.</strong> "
                "Your checklists and notes are saved to your account.</div>")
    return ("<div class='ri-mode'><strong>Demo mode.</strong> Try everything here, but nothing is stored. "
            "Create an account to save your inspections.</div>")


def history_html(entries: list[dict], signed_in: bool) -> str:
    latest, saves = {}, {}
    for entry in entries:
        latest[entry["business"]] = entry
        saves[entry["business"]] = saves.get(entry["business"], 0) + 1
    if not latest:
        message = ("No inspections saved yet. Save a checklist and the restaurant appears here." if signed_in else
                   "Nothing here yet. Save a checklist to see how this works, or create an account to keep your inspections.")
        return f"<div class='ri-card'>{message}</div>"
    cards = ""
    for name, entry in sorted(latest.items(), key=lambda item: item[1]["time"], reverse=True):
        count = f"{saves[name]} {'save' if saves[name] == 1 else 'saves'}"
        demo = "" if signed_in else "<span class='ri-tag ri-low'>demo</span>"
        tags = "".join(f"<span class='ri-tag'>{CATEGORY_LABELS[c]}: {esc(s)}</span>"
                       for c, s in entry["checklist"].items() if c in CATEGORY_LABELS)
        notes = f"<p dir='auto'>{esc(entry['notes'])}</p>" if entry.get("notes", "").strip() else ""
        cards += (f"<div class='ri-card ri-hist'><div class='ri-hist-head'><strong>{esc(name)}</strong>"
                  f"<span class='ri-muted'>Last saved {saved_date(entry)}, {count}</span></div>"
                  f"<div class='ri-tags'>{demo}{tags}</div>{notes}</div>")
    return f"<div class='ri-hist-list'>{cards}</div>"


def refresh(name: str, language: str, demo_saves: list[dict], session: dict | None):
    """Redraws everything for one business and one visitor. Runs on page load and on every new search."""
    row = business_row(name)
    session = accounts.fresh(session)
    entries = list(demo_saves or [])
    if session:
        entries = []
        try:
            entries = accounts.fetch_inspections(session)
        except accounts.AccountError as error:
            gr.Warning(str(error))
    email = session["email"] if session else None
    entry = latest_entry(name, entries)
    answers = entry["checklist"] if entry else {}
    radios = [gr.update(label=f"{CATEGORY_LABELS[c]} ({row[f'{c}_hits']} alleged)", value=answers.get(c))
              for c in CATEGORY_ORDER]
    return [risk_card(row), complaints_html(name), brief_html(name, language), sample_html(name),
            *radios, entry["notes"] if entry else "", last_saved_text(entry, bool(session)),
            gr.update(value=None, visible=False), mode_html(email), history_html(entries, bool(session)),
            gr.update(value="Sign out" if session else SIGN_IN_LABEL), session or SIGNED_OUT]


def collect_checklist(values) -> dict:
    return {category: value for category, value in zip(CATEGORY_ORDER, values) if value}


def save_checklist(name: str, notes: str, safety: str | None, hygiene: str | None, licensing: str | None,
                   pricing: str | None, service: str | None, demo_saves: list[dict], session: dict | None):
    checklist = collect_checklist((safety, hygiene, licensing, pricing, service))
    if not checklist and not notes.strip():
        raise gr.Error("Choose at least one status or add a note first.")
    if len(notes) > MAX_NOTES:
        raise gr.Error(f"Notes can be up to {MAX_NOTES} characters.")
    session = accounts.fresh(session)
    if not session:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "business": name, "checklist": checklist, "notes": notes}
        demo_saves = [*(demo_saves or []), entry]
        return ("Demo only: kept until you leave the page. Create an account to save it for good.",
                history_html(demo_saves, False), demo_saves, SIGNED_OUT)
    try:
        accounts.add_inspection(session, name, checklist, notes)
        entries = accounts.fetch_inspections(session)
    except accounts.AccountError as error:
        raise gr.Error(str(error)) from error
    return f"Saved for {name}. It is now in My inspections.", history_html(entries, True), demo_saves, session


def make_pdf(name: str, language: str, notes: str, *values):
    row = business_row(name).to_dict()
    reviews = REVIEWS[REVIEWS["business"] == name].to_dict("records")
    try:
        path = report.build_report(row, reviews, len(BUSINESSES), LANGUAGES[language], collect_checklist(values),
                                   notes, SAMPLES.get(name), TRANSLATIONS)
        return gr.update(value=path, visible=True)
    except FileNotFoundError as error:
        raise gr.Error(f"PDF font is missing: {error}") from error


def account_click(name: str, language: str, demo_saves: list[dict], session: dict | None):
    """The top button signs out, or opens the sign-in form."""
    if session:
        accounts.sign_out(session)
        return [*refresh(name, language, demo_saves, None), gr.update(visible=False), ""]
    return [*[gr.skip() for _ in range(SCREEN_SIZE)], gr.update(visible=True), ""]


def sign_in(email: str, password: str, name: str, language: str, demo_saves: list[dict]):
    return finish_auth(email, password, name, language, demo_saves, accounts.sign_in)


def sign_up(email: str, password: str, name: str, language: str, demo_saves: list[dict]):
    return finish_auth(email, password, name, language, demo_saves, accounts.sign_up)


def finish_auth(email: str, password: str, name: str, language: str, demo_saves: list[dict], action):
    """The password is passed straight to Supabase and cleared from the form, whatever the result."""
    email = (email or "").strip()
    skip = [gr.skip() for _ in range(SCREEN_SIZE)]
    if not email or not password:
        return [*skip, gr.skip(), "", "Enter your email and password.", gr.skip()]
    try:
        session = action(email, password)
    except accounts.AccountError as error:
        return [*skip, gr.skip(), "", str(error), gr.skip()]
    if not session:
        return [*skip, gr.skip(), "", "Check your email to confirm your account, then sign in.", gr.skip()]
    return [*refresh(name, language, demo_saves, session), gr.update(visible=False), "", "", ""]


def intro(text: str) -> str:
    return f"<p class='ri-section-intro'>{text}</p>"


with gr.Blocks(title="Review Inspector", fill_width=True) as demo:
    gr.HTML("<div class='ri-header'><h1>Review Inspector</h1>"
            "<p>Ranked evidence from public reviews, for inspectors in Qatar.</p></div>")

    first = BUSINESSES.iloc[0]["business"]
    demo_saves = gr.State([])
    session = gr.BrowserState(None, storage_key="review_inspector_session")
    with gr.Row(elem_id="ri-top"):
        picker = gr.Dropdown(picker_choices(), value=first, label="Search for a business", show_label=False,
                             container=False, filterable=True, elem_id="ri-search", scale=5)
        account = gr.Button(SIGN_IN_LABEL, scale=1, min_width=260, elem_id="ri-login", variant="primary")
    with gr.Group(visible=False) as auth_panel:
        email_box = gr.Textbox(label="Email", lines=1, max_lines=1, type="email")
        password_box = gr.Textbox(label="Password", lines=1, max_lines=1, type="password")
        with gr.Row(elem_classes="ri-auth-actions"):
            sign_in_button = gr.Button("Sign in", variant="primary")
            sign_up_button = gr.Button("Create account")
        auth_message = gr.Markdown()
    mode = gr.HTML(mode_html(None))
    risk = gr.HTML(risk_card(business_row(first)))

    with gr.Tabs():
        with gr.Tab("Complaints"):
            gr.HTML(intro("Safety comes first. Highlighted text is the sentence that supports the complaint. "
                          "Reviews in other languages are shown in English, with the original language in brackets."))
            complaints = gr.HTML(complaints_html(first))

        with gr.Tab("Brief and report"):
            language = gr.Radio(list(LANGUAGES), value="English", label="Report language")
            brief = gr.HTML(brief_html(first, "English"))
            sample = gr.HTML(sample_html(first))
            export = gr.Button("Create PDF report", variant="primary")
            pdf = gr.File(label="PDF report", interactive=False, visible=False)

        with gr.Tab("Checklist"):
            gr.HTML(intro("Record what you found on site. Your choices also appear in the PDF report."))
            radios = [gr.Radio(STATUSES, elem_classes="ri-check",
                               label=f"{CATEGORY_LABELS[c]} ({business_row(first)[f'{c}_hits']} alleged)")
                      for c in CATEGORY_ORDER]
            notes = gr.Textbox(label="Notes", lines=3)
            save = gr.Button("Save checklist", variant="primary")
            saved = gr.Markdown()

        with gr.Tab("My inspections"):
            gr.HTML(intro("Restaurants with a saved checklist, newest first. Search a restaurant above to reopen it."))
            history = gr.HTML(history_html([], False))

        with gr.Tab("All businesses"):
            gr.HTML(ranking_html())

        with gr.Tab("How this works"):
            gr.Markdown(METHOD)

    screen = [risk, complaints, brief, sample, *radios, notes, saved, pdf, mode, history, account, session]
    picker.change(refresh, [picker, language, demo_saves, session], screen, api_name="show_business")
    demo.load(refresh, [picker, language, demo_saves, session], screen, api_name=False)
    language.change(brief_html, [picker, language], brief, api_name="brief")
    save.click(save_checklist, [picker, notes, *radios, demo_saves, session], [saved, history, demo_saves, session],
               api_name="save_checklist")
    export.click(make_pdf, [picker, language, notes, *radios], pdf, api_name="make_pdf")
    account.click(account_click, [picker, language, demo_saves, session], [*screen, auth_panel, auth_message], api_name=False)
    form = [email_box, password_box, picker, language, demo_saves]
    after = [*screen, auth_panel, password_box, auth_message, email_box]
    sign_in_button.click(sign_in, form, after, api_name=False)
    sign_up_button.click(sign_up, form, after, api_name=False)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=FONTS, ssr_mode=False)
