"""Review Inspector: reads precomputed results and works in English or Arabic."""
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
import records
import report
from i18n import CHOICES, STATUSES, UI, account_error

DATASET_REPO = "Hermesione/reviewere-dataset"
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR")
MAX_SHOWN = {"en": 6, "ar": report.MAX_EXAMPLES}  # Arabic translations cover the reviews reports show.
MIN_SENTENCE_MATCH = 0.6
MAX_NOTES = 5000
DEFAULT_LANGUAGE = "English"

CATEGORY_ORDER = report.CATEGORY_ORDER
INSPECTED_ON = "inspected_on"  # Checklist key that marks an uploaded past record and holds the date of the visit.
SIGNED_OUT = {}  # Browser storage keeps its old value if given None, so sign-out must store something empty.
PROFILE_ICON = ("<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
                "stroke-linecap='round' aria-hidden='true'><circle cx='12' cy='8' r='4'/><path d='M4 21a8 8 0 0 1 16 0'/></svg>")
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
         "<script>document.addEventListener('focusin',e=>{if(e.target.matches('#ri-search input'))e.target.select()})</script>"
         # Pop-up windows close on Escape or a click outside them.
         "<script>document.addEventListener('keydown',e=>{if(e.key==='Escape')document.querySelectorAll('.ri-modal-close')"
         ".forEach(b=>{if(b.offsetParent)b.click()})});"
         "document.addEventListener('click',e=>{if(e.target.classList?.contains('ri-modal'))"
         "e.target.querySelector('.ri-modal-close').click()})</script>")
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
#ri-search input{height:48px;padding-inline-start:44px!important;font-size:1rem;background:var(--input-background-fill) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%2371717a' stroke-width='2.2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.8-3.8'/%3E%3C/svg%3E") no-repeat 15px center}
#ri-search .icon-wrap,#ri-search svg.dropdown-arrow{display:none!important}
#ri-search,#ri-search .wrap,#ri-search .wrap-inner{padding:0!important;min-height:0;background:transparent!important;border:none!important;box-shadow:none!important}
#ri-search input{border:1px solid var(--input-border-color);border-radius:10px;width:100%}
.ri-stars{position:relative;display:inline-block;white-space:nowrap;line-height:1;letter-spacing:1px;font-size:1rem;font-weight:400;color:var(--neutral-400);font-family:system-ui,sans-serif;vertical-align:baseline}
.ri-stars span{position:absolute;top:0;left:0;bottom:0;overflow:hidden;color:#f5a400}
.ri-tag .ri-stars{font-size:.92rem}
#ri-head{align-items:flex-start;gap:16px}
body:has(#huggingface-space-header) #ri-head{padding-top:46px}
html[dir=rtl] .block{direction:rtl}
html[dir=rtl] .prose,html[dir=rtl] textarea{direction:rtl;text-align:right}
html[dir=rtl] input[type=email],html[dir=rtl] input[type=password]{direction:ltr}
html[dir=rtl] label:has(input[type=radio])>span{margin-left:0;margin-right:var(--size-2)}
#ri-language{flex-grow:0;background:transparent!important;border:none!important;padding:0!important}
#ri-language .wrap{flex-wrap:nowrap;gap:6px}
#ri-language label{padding:10px 12px;white-space:nowrap}
html[dir=rtl] #ri-search input{background-position:right 15px center!important}
.ri-stars{direction:ltr}
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
#ri-top{align-items:center;gap:14px}
#ri-login{height:48px;white-space:nowrap}
.ri-auth-actions{gap:10px}
#ri-head{position:relative;z-index:50;overflow:visible}
#ri-profile{position:relative;flex:0 0 44px;width:44px;overflow:visible}
.ri-avatar{display:grid;place-items:center;width:44px;height:44px;border-radius:50%;border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);color:var(--body-text-color);cursor:default}
.ri-avatar:focus-visible{outline:2px solid var(--color-accent);outline-offset:2px}
#ri-profile-card{position:absolute;top:calc(100% + 8px);inset-inline-end:0;width:290px;max-width:calc(100vw - 32px);gap:12px;padding:16px 18px;border:1px solid var(--border-color-primary);border-radius:12px;background:var(--background-fill-primary);box-shadow:0 10px 30px rgba(0,0,0,.18);visibility:hidden;opacity:0;transition:opacity .12s}
#ri-profile-card::before{content:'';position:absolute;inset-inline:0;top:-10px;height:10px}
#ri-profile:hover #ri-profile-card,#ri-profile:focus-within #ri-profile-card{visibility:visible;opacity:1}
.ri-profile-info{display:flex;flex-direction:column;gap:4px}
.ri-profile-info strong{overflow-wrap:anywhere;text-align:start}
.ri-modal{position:fixed!important;inset:0;z-index:1000;justify-content:center!important;align-items:center!important;padding:16px!important;background:rgba(0,0,0,.55)}
.ri-modal-box{position:relative;flex:0 0 auto!important;width:420px!important;max-width:100%;max-height:100%;overflow-y:auto;gap:14px!important;padding:24px!important;border:1px solid var(--border-color-primary);border-radius:12px;background:var(--background-fill-primary);box-shadow:0 20px 50px rgba(0,0,0,.3)}
.ri-modal-close{position:absolute!important;top:12px;inset-inline-end:12px;z-index:1;width:32px!important;min-width:32px!important;height:32px;padding:0!important;font-size:1.3rem;line-height:1;border:none!important;background:transparent!important;box-shadow:none!important}
#ri-auth{justify-content:flex-start!important;padding-top:clamp(56px,14vh,140px)!important}
.ri-modal-title{margin:0;padding-inline-end:32px;font-size:1.2rem;font-weight:600}
#ri-past{align-items:center;flex-wrap:nowrap;gap:16px;border:1px solid var(--border-color-primary);background:var(--background-fill-secondary);border-radius:12px;padding:18px 24px}
#ri-past p{margin:0;line-height:1.7;overflow-wrap:anywhere}
#ri-upload{flex:0 0 auto;width:auto;white-space:nowrap}
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
  #ri-head{flex-direction:column!important;align-items:stretch;gap:12px;padding-top:46px}
  #ri-profile{position:absolute;bottom:0;inset-inline-end:0}
  #ri-past{flex-direction:column;align-items:stretch;padding:18px}
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
TRANSLATIONS_AR = {row.text: row.text_ar for row in pd.read_parquet(data_path("translations_ar.parquet")).itertuples()
                   if isinstance(row.text_ar, str)}
SAMPLES = {r["business"]: r for r in json.loads(Path(data_path("mock_inspections.json")).read_text(encoding="utf-8"))["records"]}


def esc(value) -> str:
    """Review text is untrusted, so it is always escaped before going into HTML."""
    return html.escape(str(value))


def stars_html(value: float, code: str) -> str:
    """Google-style stars. The gold layer is cut to size, so 3.9 fills 78% of the row."""
    label = UI[code]["stars_label"].format(value=value)
    return (f"<span class='ri-stars' role='img' aria-label='{label}' title='{label}'>"
            f"<span style='width:{value / 5 * 100:.0f}%'>★★★★★</span>★★★★★</span>")


def business_row(name: str) -> pd.Series:
    return BUSINESSES[BUSINESSES["business"] == name].iloc[0]


def shown_name(name: str, code: str) -> str:
    return TRANSLATIONS_AR.get(name, name) if code == "ar" else name


def picker_choices(code: str) -> list[tuple[str, str]]:
    return [(f"{r['rank']}. {shown_name(r['business'], code)}", r["business"]) for _, r in BUSINESSES.iterrows()]


def ranking_html(code: str) -> str:
    text = UI[code]
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{esc(shown_name(r['business'], code))}</td><td>{esc(text['kinds'][r['category']])}</td>"
        f"<td>{r['risk_score']:.2f}</td><td>{r['place_rating']} {stars_html(r['place_rating'], code)}</td>"
        f"<td>{r['reviews_scored']}</td></tr>"
        for _, r in BUSINESSES.iterrows()
    )
    head = "".join(f"<th>{label}</th>" for label in text["table"])
    return (f"<div class='ri-card ri-table-wrap'><table class='ri-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def level_of(row: pd.Series, category: str) -> str:
    """Compares a business with the others in one category. The levels are relative, not absolute."""
    if row[f"{category}_hits"] == 0:
        return "None"
    position = int((BUSINESSES[f"{category}_severity"] > row[f"{category}_severity"]).sum()) + 1
    return "High" if position <= 3 else "Medium" if position <= 6 else "Low"


def count_text(code: str, count: int) -> str:
    return UI[code]["one_review"] if count == 1 else UI[code]["many_reviews"].format(count=count)


def category_row(row: pd.Series, category: str, code: str) -> str:
    text = UI[code]
    hits, credible = row[f"{category}_hits"], row[f"{category}_credible_hits"]
    level = level_of(row, category)
    reliable = text["reliable"].format(count=credible) if hits else text["nothing_to_check"]
    return (f"<div class='ri-row'><div class='ri-row-main'><strong>{text['categories'][category]}</strong>"
            f"<span class='ri-muted'>{text['category_help'][category]}</span></div>"
            f"<div class='ri-row-count'><strong>{count_text(code, hits)}</strong><span class='ri-muted'>{reliable}</span></div>"
            f"<span class='ri-level ri-level-{level.lower()}'>{text['levels'][level]}</span></div>")


def risk_card(row: pd.Series, code: str) -> str:
    text = UI[code]
    rows = "".join(category_row(row, category, code) for category in CATEGORY_ORDER)
    stats = (
        (text["priority_rank"], text["rank_of"].format(rank=row["rank"], total=len(BUSINESSES))),
        (text["priority_score"], f"{row['risk_score']:.2f}"),
        (text["google_rating"], f"{row['place_rating']} {stars_html(row['place_rating'], code)}"),
        (text["reviews_analysed"], row["reviews_scored"]),
        (text["average_credibility"], f"{row['mean_credibility']:.2f}"),
    )
    tiles = "".join(f"<div class='ri-stat'><span class='ri-muted'>{label}</span><b>{value}</b></div>"
                    for label, value in stats)
    place = text["place_line"].format(kind=text["kinds"][row["category"]].capitalize(), count=row["place_reviews"])
    return f"""<div class='ri-card'>
      <div class='ri-title'><h2>{esc(shown_name(row['business'], code))}</h2>
      <span class='ri-muted'>{esc(place)}</span></div>
      <div class='ri-stats'>{tiles}</div>
      <div class='ri-title'><p class='ri-subhead'>{text['complaining']}</p>
      <span class='ri-muted'>{text['concern_help'].format(others=len(BUSINESSES) - 1)}</span></div>
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


def english_body(review: pd.Series, evidence: str) -> str:
    english = TRANSLATIONS.get(review["text"])
    if not english:
        return f"<p dir='auto'>{highlight(review['text'], evidence)}</p>"
    return (f"<p dir='ltr' lang='en'>{highlight_translation(english, TRANSLATIONS.get(evidence))}</p>"
            f"<p class='ri-translated-from'>({esc(report.translated_from(review['text'], review['language']))})</p>")


def arabic_body(review: pd.Series, evidence: str) -> tuple[str, bool]:
    """Returns the review in Arabic, and whether only its key sentence was available."""
    text = review["text"]
    if report.mostly_arabic(text):
        return f"<p dir='rtl' lang='ar'>{highlight(text, evidence)}</p>", False
    arabic = TRANSLATIONS_AR.get(text) or TRANSLATIONS_AR.get(evidence)
    if not arabic:
        return f"<p dir='auto'>{esc(text)}</p>", False
    label = report.translated_from_arabic_label(review["language"])
    body = f"<p dir='rtl' lang='ar'>{esc(arabic)}</p><p class='ri-translated-from' dir='rtl'>({esc(label)})</p>"
    return body, text not in TRANSLATIONS_AR and evidence != text


def review_html(review: pd.Series, category: str, code: str) -> str:
    text = UI[code]
    low = review[f"{category}_low_confidence"]
    praise = review[f"{category}_likely_praise"]
    evidence = review[f"{category}_evidence"]
    if code == "ar":
        body, key_only = arabic_body(review, evidence)
    else:
        body, key_only = english_body(review, evidence), False
    date = f"{review['timestamp_utc']:%Y-%m}" if code == "ar" else f"{review['timestamp_utc']:%b %Y}"
    tags = [
        f"<span class='ri-tag'>{stars_html(review['rating'], code)}</span>",
        f"<span class='ri-tag'>{date}</span>",
        f"<span class='ri-tag'>{text['confidence'].format(value=review[f'{category}_confidence'])}</span>",
        f"<span class='ri-tag'>{text['credibility'].format(value=review['credibility'])}</span>",
        f"<span class='ri-tag ri-low'>{text['likely_praise']}</span>" if praise else "",
        f"<span class='ri-tag ri-low'>{text['low_confidence']}</span>" if low and not praise else "",
        f"<span class='ri-tag'>{text['whole_review']}</span>" if evidence == review["text"] else "",
        f"<span class='ri-tag'>{text['key_sentence']}</span>" if key_only else "",
    ]
    return f"<div class='ri-review'><div class='ri-tags'>{''.join(tags)}</div>{body}</div>"


def complaints_html(name: str, code: str) -> str:
    text = UI[code]
    group = REVIEWS[REVIEWS["business"] == name]
    sections = ""
    for category in CATEGORY_ORDER:
        hits = group[group[f"{category}_confidence"].notna()]
        if hits.empty:
            continue
        order = [f"{category}_likely_praise", f"{category}_low_confidence", f"{category}_confidence"]
        ranked = hits.sort_values(order, ascending=[True, True, False])
        praise = int(hits[f"{category}_likely_praise"].sum())
        extra = text["praise_extra"].format(count=praise) if praise else ""
        body = "".join(review_html(review, category, code) for _, review in ranked.head(MAX_SHOWN[code]).iterrows())
        hidden = len(hits) - MAX_SHOWN[code]
        more = f"<div class='ri-muted ri-more'>{text['not_shown'].format(count=hidden)}</div>" if hidden > 0 else ""
        sections += (f"<details class='ri-group'><summary>{text['categories'][category]} ({len(hits) - praise}{extra})</summary>"
                     f"{body}{more}</details>")
    if not sections:
        return f"<div class='ri-card'>{text['no_complaints']}</div>"
    return f"<div class='ri-groups'>{sections}</div>"


def brief_html(name: str, language: str) -> str:
    code = CHOICES[language]
    row = business_row(name)
    source = str(row[f"brief_source_{code}"])
    note = "" if source.startswith("llm") else f"<span class='ri-muted'>{UI[code]['written_by_template']}</span>"
    direction = "rtl" if code == "ar" else "ltr"
    written = report.arabic_brief(row["brief_ar"]) if code == "ar" else row["brief_en"]
    return (f"<div class='ri-stack'><div class='ri-card' dir='{direction}' lang='{code}'>"
            f"<h3 class='ri-card-title'>{report.TEXT[code]['brief']}</h3><p class='ri-brief'>"
            f"{esc(written)}</p></div>{note}</div>")


def inspected_on(entry: dict) -> str:
    """Uploaded records keep the date of the visit in their checklist. Saved checklists use the day they were saved."""
    return (entry["checklist"].get(INSPECTED_ON) or entry["time"])[:10]


def previous_inspection(name: str, entries: list[dict]) -> dict | None:
    return max((entry for entry in entries if entry["business"] == name), key=inspected_on, default=None)


def past_html(name: str, entries: list[dict], code: str) -> str:
    text = UI[code]
    previous = previous_inspection(name, entries)
    if previous:
        notes = f"<p class='ri-muted' dir='auto'>{esc(previous['notes'])}</p>" if previous.get("notes", "").strip() else ""
        return f"<div class='ri-stack'><p><strong>{text['previous_on_record'].format(date=inspected_on(previous))}</strong></p>{notes}</div>"
    sample = SAMPLES.get(name)
    if not sample:
        return f"<p>{text['no_previous']}</p>"
    visit = text["visit_types"].get(sample["visit_type"], sample["visit_type"])
    outcome = text["outcomes"].get(sample["outcome"], sample["outcome"])
    line = text["previous_inspection"].format(date=esc(sample["last_visit"]), visit=esc(visit), outcome=esc(outcome))
    return f"<div class='ri-stack'><div class='ri-tags'><span class='ri-tag ri-sample'>{text['sample_tag']}</span></div><p>{line}</p></div>"


def saved_date(entry: dict, code: str) -> str:
    return datetime.fromisoformat(inspected_on(entry)).strftime("%Y-%m-%d" if code == "ar" else "%d %b %Y")


def latest_entry(name: str, entries: list[dict]) -> dict | None:
    """The last checklist saved here. Uploaded records have nothing to restore into the form."""
    return next((entry for entry in reversed(entries) if entry["business"] == name and INSPECTED_ON not in entry["checklist"]), None)


def last_saved_text(entry: dict | None, signed_in: bool, code: str) -> str:
    if not entry:
        return ""
    if not signed_in:
        return UI[code]["demo_kept"]
    return UI[code]["restored"].format(date=saved_date(entry, code))


def mode_html(code: str) -> str:
    return f"<div class='ri-mode'>{UI[code]['mode_demo']}</div>"


def avatar_html(code: str) -> str:
    return f"<div class='ri-avatar' tabindex='0' aria-label='{UI[code]['signed_in']}'>{PROFILE_ICON}</div>"


def profile_html(email: str | None, code: str) -> str:
    text = UI[code]
    return (f"<div class='ri-profile-info'><span class='ri-muted'>{text['signed_in']}</span>"
            f"<strong dir='ltr'>{esc(email or '')}</strong><span class='ri-muted'>{text['account_saved']}</span></div>")


def modal_title(title: str) -> str:
    return f"<h2 class='ri-modal-title'>{title}</h2>"


def history_html(entries: list[dict], signed_in: bool, code: str) -> str:
    text = UI[code]
    latest, saves = {}, {}
    for entry in entries:
        latest[entry["business"]] = entry
        saves[entry["business"]] = saves.get(entry["business"], 0) + 1
    if not latest:
        return f"<div class='ri-card'>{text['history_empty_signed_in' if signed_in else 'history_empty_demo']}</div>"
    cards = ""
    for name, entry in sorted(latest.items(), key=lambda item: item[1]["time"], reverse=True):
        count = text["one_save"] if saves[name] == 1 else text["many_saves"].format(count=saves[name])
        demo = "" if signed_in else f"<span class='ri-tag ri-low'>{text['demo_tag']}</span>"
        tags = "".join(f"<span class='ri-tag'>{text['categories'][c]}: {esc(text['statuses'].get(s, s))}</span>"
                       for c, s in entry["checklist"].items() if c in text["categories"])
        notes = f"<p dir='auto'>{esc(entry['notes'])}</p>" if entry.get("notes", "").strip() else ""
        saved_line = text["last_saved"].format(date=saved_date(entry, code), saves=count)
        cards += (f"<div class='ri-card ri-hist'><div class='ri-hist-head'><strong>{esc(shown_name(name, code))}</strong>"
                  f"<span class='ri-muted'>{saved_line}</span></div>"
                  f"<div class='ri-tags'>{demo}{tags}</div>{notes}</div>")
    return f"<div class='ri-hist-list'>{cards}</div>"


def intro(text: str) -> str:
    return f"<p class='ri-section-intro'>{text}</p>"


def header_html(code: str) -> str:
    return f"<div class='ri-header'><h1>{UI[code]['title']}</h1><p>{UI[code]['subtitle']}</p></div>"


def status_choices(code: str) -> list[tuple[str, str]]:
    return [(UI[code]["statuses"][status], status) for status in STATUSES]


def collect_checklist(values) -> dict:
    return {category: value for category, value in zip(CATEGORY_ORDER, values) if value}


def load_entries(session: dict | None, demo_saves: list[dict], code: str) -> list[dict]:
    if not session:
        return list(demo_saves or [])
    try:
        return accounts.fetch_inspections(session)
    except accounts.AccountError as error:
        gr.Warning(account_error(code, str(error)))
        return []


def draw(name: str, report_language: str, demo_saves: list[dict], session: dict | None, site: str, report_value=None):
    """Redraws everything for one business, one visitor and one site language."""
    code = CHOICES[site]
    text = UI[code]
    row = business_row(name)
    session = accounts.fresh(session)
    entries = load_entries(session, demo_saves, code)
    entry = latest_entry(name, entries)
    answers = entry["checklist"] if entry else {}
    radios = [gr.update(label=text["alleged"].format(label=text["categories"][c], count=row[f"{c}_hits"]),
                        value=answers.get(c), choices=status_choices(code)) for c in CATEGORY_ORDER]
    report_update = gr.update(label=text["report_language"]) if report_value is None else gr.update(label=text["report_language"], value=report_value)
    return [
        risk_card(row, code), complaints_html(name, code), brief_html(name, report_value or report_language), past_html(name, entries, code),
        *radios, gr.update(label=text["notes"], value=entry["notes"] if entry else ""),
        last_saved_text(entry, bool(session), code),
        gr.update(value=mode_html(code), visible=not session), history_html(entries, bool(session), code),
        gr.update(value=text["open_account"], visible=not session), session or SIGNED_OUT,
        header_html(code), gr.update(choices=picker_choices(code), label=text["search"]), report_update, intro(text["intro_complaints"]),
        intro(text["intro_checklist"]), intro(text["intro_inspections"]), ranking_html(code), text["method"],
        gr.update(value=text["save"]), gr.update(value=text["create_pdf"]), gr.update(label=text["email"]),
        gr.update(label=text["password"]), gr.update(value=text["sign_in"]), gr.update(value=text["create_account"]),
        *[gr.update(label=text[key]) for key in ("tab_complaints", "tab_brief", "tab_checklist", "tab_inspections", "tab_businesses", "tab_method")],
        gr.update(visible=bool(session)), avatar_html(code), profile_html(session["email"] if session else None, code),
        gr.update(value=text["sign_out"]), modal_title(text["open_account"]),
        gr.update(label=text["upload_record"]),
    ]


def show_business(name: str, report_language: str, demo_saves: list[dict], session: dict | None, site: str):
    return draw(name, report_language, demo_saves, session, site)


def open_page(name: str, demo_saves: list[dict], session: dict | None, stored: str | None):
    """Applies the visitor's saved language when the page opens."""
    site = stored if stored in CHOICES else DEFAULT_LANGUAGE
    return [gr.update(value=site, label=UI[CHOICES[site]]["site_language"]), *draw(name, site, demo_saves, session, site, report_value=site)]


def switch_language(name: str, demo_saves: list[dict], session: dict | None, site: str):
    """The report follows the site language, and can still be changed in its own tab."""
    return [*draw(name, site, demo_saves, session, site, report_value=site), site, gr.update(label=UI[CHOICES[site]]["site_language"])]


def save_checklist(name: str, notes: str, safety: str | None, hygiene: str | None, licensing: str | None,
                   pricing: str | None, service: str | None, demo_saves: list[dict], session: dict | None, site: str):
    code = CHOICES[site]
    text = UI[code]
    checklist = collect_checklist((safety, hygiene, licensing, pricing, service))
    if not checklist and not notes.strip():
        raise gr.Error(text["need_status"])
    if len(notes) > MAX_NOTES:
        raise gr.Error(text["notes_too_long"].format(limit=MAX_NOTES))
    session = accounts.fresh(session)
    if not session:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "business": name, "checklist": checklist, "notes": notes}
        demo_saves = [*(demo_saves or []), entry]
        return text["demo_saved"], history_html(demo_saves, False, code), demo_saves, SIGNED_OUT, past_html(name, demo_saves, code)
    try:
        accounts.add_inspection(session, name, checklist, notes)
        entries = accounts.fetch_inspections(session)
    except accounts.AccountError as error:
        raise gr.Error(account_error(code, str(error))) from error
    return text["saved"].format(name=shown_name(name, code)), history_html(entries, True, code), demo_saves, session, past_html(name, entries, code)


def upload_record(path: str, name: str, demo_saves: list[dict], session: dict | None, site: str):
    """Reads a past record with the text model, then files it like any other inspection."""
    code = CHOICES[site]
    text = UI[code]
    try:
        record = records.parse_record(path, "Arabic" if code == "ar" else "English")
    except records.RecordError as error:
        raise gr.Error(text[str(error)]) from error
    checklist = {INSPECTED_ON: record["inspected_on"]}
    session = accounts.fresh(session)
    if session:
        try:
            accounts.add_inspection(session, name, checklist, record["summary"])
            entries = accounts.fetch_inspections(session)
        except accounts.AccountError as error:
            raise gr.Error(account_error(code, str(error))) from error
    else:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "business": name, "checklist": checklist, "notes": record["summary"]}
        demo_saves = entries = [*(demo_saves or []), entry]
    gr.Info(text["record_added"].format(date=record["inspected_on"]))
    return past_html(name, entries, code), history_html(entries, bool(session), code), demo_saves, session or SIGNED_OUT


def make_pdf(name: str, language: str, demo_saves: list[dict], session: dict | None, notes: str, *values):
    row = business_row(name).to_dict()
    reviews = REVIEWS[REVIEWS["business"] == name].to_dict("records")
    code = CHOICES[language]
    previous = previous_inspection(name, load_entries(accounts.fresh(session), demo_saves, code))
    try:
        path = report.build_report(row, reviews, len(BUSINESSES), code, collect_checklist(values), notes,
                                   SAMPLES.get(name), {"en": TRANSLATIONS, "ar": TRANSLATIONS_AR}[code],
                                   {"date": inspected_on(previous), "notes": previous["notes"]} if previous else None)
        return path
    except FileNotFoundError as error:
        raise gr.Error(f"PDF font is missing: {error}") from error


def open_auth():
    return gr.update(visible=True), ""


def close_auth():
    return gr.update(visible=False), "", ""


def sign_out(name: str, report_language: str, demo_saves: list[dict], session: dict | None, site: str):
    if session:
        accounts.sign_out(session)
    return draw(name, report_language, demo_saves, None, site)


def sign_in(email: str, password: str, name: str, report_language: str, demo_saves: list[dict], site: str):
    return finish_auth(email, password, name, report_language, demo_saves, site, accounts.sign_in)


def sign_up(email: str, password: str, name: str, report_language: str, demo_saves: list[dict], site: str):
    return finish_auth(email, password, name, report_language, demo_saves, site, accounts.sign_up)


def finish_auth(email: str, password: str, name: str, report_language: str, demo_saves: list[dict], site: str, action):
    """The password is passed straight to Supabase and cleared from the form, whatever the result."""
    code = CHOICES[site]
    email = (email or "").strip()
    skip = [gr.skip() for _ in range(DRAW_SIZE)]
    if not email or not password:
        return [*skip, gr.skip(), "", UI[code]["enter_details"], gr.skip()]
    try:
        session = action(email, password)
    except accounts.AccountError as error:
        return [*skip, gr.skip(), "", account_error(code, str(error)), gr.skip()]
    if not session:
        return [*skip, gr.skip(), "", UI[code]["confirm_email"], gr.skip()]
    return [*draw(name, report_language, demo_saves, session, site), gr.update(visible=False), "", "", ""]


SET_DIRECTION = """(site) => {
  const arabic = site === 'العربية';
  document.documentElement.dir = arabic ? 'rtl' : 'ltr';
  document.documentElement.lang = arabic ? 'ar' : 'en';
}"""

SAVE_FILE = """(file) => {
  if (!file?.url) return;
  const link = document.createElement('a');
  link.href = file.url;
  link.download = file.orig_name;
  document.body.append(link);
  link.click();
  link.remove();
}"""

with gr.Blocks(title="Review Inspector", fill_width=True) as demo:
    text = UI["en"]
    first = BUSINESSES.iloc[0]["business"]
    first_row = business_row(first)
    with gr.Row(elem_id="ri-head"):
        header = gr.HTML(header_html("en"), scale=1)
        site_language = gr.Radio(list(CHOICES), value=DEFAULT_LANGUAGE, label=text["site_language"], show_label=False,
                                 container=False, elem_id="ri-language", scale=0, min_width=190)
        with gr.Column(visible=False, elem_id="ri-profile", scale=0, min_width=44) as profile:
            avatar = gr.HTML(avatar_html("en"))
            with gr.Column(elem_id="ri-profile-card", min_width=0):
                profile_info = gr.HTML(profile_html(None, "en"))
                sign_out_button = gr.Button(text["sign_out"], size="sm")

    demo_saves = gr.State([])
    session = gr.BrowserState(None, storage_key="review_inspector_session")
    stored_language = gr.BrowserState(DEFAULT_LANGUAGE, storage_key="review_inspector_language")
    with gr.Row(elem_id="ri-top"):
        picker = gr.Dropdown(picker_choices("en"), value=first, label=text["search"], show_label=False,
                             container=False, filterable=True, elem_id="ri-search", scale=5)
        account = gr.Button(text["open_account"], scale=1, min_width=260, elem_id="ri-login", variant="primary")
    with gr.Column(visible=False, elem_id="ri-auth", elem_classes="ri-modal") as auth_panel:
        with gr.Column(elem_classes="ri-modal-box", min_width=0):
            auth_close = gr.Button("×", elem_classes="ri-modal-close")
            auth_title = gr.HTML(modal_title(text["open_account"]))
            email_box = gr.Textbox(label=text["email"], lines=1, max_lines=1, type="email")
            password_box = gr.Textbox(label=text["password"], lines=1, max_lines=1, type="password")
            with gr.Row(elem_classes="ri-auth-actions"):
                sign_in_button = gr.Button(text["sign_in"], variant="primary")
                sign_up_button = gr.Button(text["create_account"])
            auth_message = gr.Markdown()
    mode = gr.HTML(mode_html("en"))
    risk = gr.HTML(risk_card(first_row, "en"))

    with gr.Tabs():
        with gr.Tab(text["tab_complaints"]) as tab_complaints:
            complaints_intro = gr.HTML(intro(text["intro_complaints"]))
            complaints = gr.HTML(complaints_html(first, "en"))

        with gr.Tab(text["tab_brief"]) as tab_brief:
            report_language = gr.Radio(list(CHOICES), value=DEFAULT_LANGUAGE, label=text["report_language"])
            brief = gr.HTML(brief_html(first, DEFAULT_LANGUAGE))
            with gr.Row(elem_id="ri-past"):
                sample = gr.HTML(past_html(first, [], "en"), scale=1)
                upload = gr.UploadButton(text["upload_record"], file_types=records.FILE_TYPES, size="sm", scale=0,
                                         min_width=150, elem_id="ri-upload")
            export = gr.Button(text["create_pdf"], variant="primary")
            pdf = gr.DownloadButton(visible="hidden")

        with gr.Tab(text["tab_checklist"]) as tab_checklist:
            checklist_intro = gr.HTML(intro(text["intro_checklist"]))
            radios = [gr.Radio(status_choices("en"), elem_classes="ri-check",
                               label=text["alleged"].format(label=text["categories"][c], count=first_row[f"{c}_hits"]))
                      for c in CATEGORY_ORDER]
            notes = gr.Textbox(label=text["notes"], lines=3)
            save = gr.Button(text["save"], variant="primary")
            saved = gr.Markdown()

        with gr.Tab(text["tab_inspections"]) as tab_inspections:
            inspections_intro = gr.HTML(intro(text["intro_inspections"]))
            history = gr.HTML(history_html([], False, "en"))

        with gr.Tab(text["tab_businesses"]) as tab_businesses:
            ranking = gr.HTML(ranking_html("en"))

        with gr.Tab(text["tab_method"]) as tab_method:
            method = gr.Markdown(text["method"])

    screen = [risk, complaints, brief, sample, *radios, notes, saved, mode, history, account, session, header, picker,
              report_language, complaints_intro, checklist_intro, inspections_intro, ranking, method, save, export, email_box,
              password_box, sign_in_button, sign_up_button, tab_complaints, tab_brief, tab_checklist, tab_inspections,
              tab_businesses, tab_method, profile, avatar, profile_info, sign_out_button, auth_title, upload]
    DRAW_SIZE = len(screen)

    picker.change(show_business, [picker, report_language, demo_saves, session, site_language], screen, api_name="show_business")
    demo.load(open_page, [picker, demo_saves, session, stored_language], [site_language, *screen], api_name=False).then(
        None, site_language, None, js=SET_DIRECTION)
    site_language.input(switch_language, [picker, demo_saves, session, site_language], [*screen, stored_language, site_language],
                        api_name=False).then(None, site_language, None, js=SET_DIRECTION)
    report_language.input(brief_html, [picker, report_language], brief, api_name="brief")
    save.click(save_checklist, [picker, notes, *radios, demo_saves, session, site_language],
               [saved, history, demo_saves, session, sample], api_name="save_checklist")
    export.click(make_pdf, [picker, report_language, demo_saves, session, notes, *radios], pdf,
                 api_name="make_pdf", show_progress_on=export).success(None, pdf, None, js=SAVE_FILE)
    upload.upload(upload_record, [upload, picker, demo_saves, session, site_language], [sample, history, demo_saves, session],
                  api_name=False, show_progress_on=upload)
    account.click(open_auth, None, [auth_panel, auth_message], api_name=False)
    auth_close.click(close_auth, None, [auth_panel, password_box, auth_message], api_name=False)
    sign_out_button.click(sign_out, [picker, report_language, demo_saves, session, site_language], screen, api_name=False)
    form = [email_box, password_box, picker, report_language, demo_saves, site_language]
    after = [*screen, auth_panel, password_box, auth_message, email_box]
    sign_in_button.click(sign_in, form, after, api_name=False)
    sign_up_button.click(sign_up, form, after, api_name=False)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, head=FONTS, ssr_mode=False)
