"""PDF inspection report for one business, in English or Arabic."""
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import Align
from fpdf.fonts import FontFace

BRAND_FONTS = Path(__file__).parent / "fonts"
FALLBACK_FONTS = Path(os.environ.get("REPORT_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
MAROON = (138, 21, 56)
GOLD = (222, 146, 0)
TRANSLATION = (84, 84, 92)
RULE = (214, 214, 219)
TABLE_HEAD = (246, 240, 242)
TABLE_WIDTHS = (24, 19, 27, 108)
CATEGORY_ORDER = ("safety", "hygiene", "licensing", "pricing_fraud", "service_only")
MAX_EXAMPLES = 5
MAX_EXAMPLE_CHARS = 280
MAX_TRANSLATION_CHARS = 340
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French", "hi": "Hindi", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "nb": "Norwegian", "sv": "Swedish", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh": "Chinese",
}

BRIEF_NAMES_AR = {"Le Park": "لو بارك", "Tourist Hotel": "السائح", "Desert Rose": "وردة الصحراء"}  # Left in Latin letters by the brief model.
ARABIC_LANGUAGE_NAMES = {
    "de": "الألمانية", "en": "الإنجليزية", "es": "الإسبانية", "fr": "الفرنسية", "hi": "الهندية", "id": "الإندونيسية",
    "it": "الإيطالية", "ja": "اليابانية", "ko": "الكورية", "ms": "الملايوية", "nb": "النرويجية", "nl": "الهولندية",
    "pl": "البولندية", "pt": "البرتغالية", "ru": "الروسية", "sv": "السويدية", "th": "التايلاندية", "tl": "التاغالوغية",
    "tr": "التركية", "uk": "الأوكرانية", "ur": "الأردية", "vi": "الفيتنامية", "zh": "الصينية",
}

TEXT = {
    "en": {
        "title": "Review Inspector report",
        "summary": "Summary",
        "rank": "Priority rank",
        "of": "of",
        "risk": "Priority score",
        "rating": "Google rating",
        "reviews": "Reviews analysed",
        "credibility": "Average credibility",
        "sample_note": "The sample deliberately includes each business's lowest-rated reviews, "
                       "so counts are not percentages of all reviews.",
        "complaints": "Alleged complaints",
        "none": "No complaints in this category.",
        "brief": "Brief",
        "brief_template": "This brief came from the fixed template, not the AI model.",
        "evidence": "Evidence from reviews",
        "low": "low confidence",
        "columns": ("Rating", "Date", "Confidence", "What the review says"),
        "checklist": "Inspector checklist",
        "unset": "Not reviewed",
        "notes": "Notes",
        "previous_title": "Previous inspection",
        "sample_title": "Previous inspection (sample data)",
        "sample_warning": "Sample data for demonstration only. This is not a real inspection record.",
        "generated": "Generated",
        "timezone": "UTC",
        "categories": {"safety": "Safety", "hygiene": "Hygiene", "licensing": "Licensing",
                       "pricing_fraud": "Pricing", "service_only": "Service only"},
        "statuses": {"Confirmed": "Confirmed", "Ongoing": "Ongoing", "False alarm": "False alarm", "Resolved": "Resolved"},
    },
    "ar": {
        "title": "تقرير مفتش المراجعات",
        "summary": "الملخص",
        "rank": "ترتيب الأولوية",
        "of": "من",
        "risk": "درجة الأولوية",
        "rating": "تقييم خرائط جوجل",
        "reviews": "المراجعات المحللة",
        "credibility": "متوسط المصداقية",
        "sample_note": "تتضمن العينة عمدا أقل المراجعات تقييما لكل منشأة، لذلك فإن الأعداد ليست نسبا من جميع المراجعات.",
        "complaints": "الشكاوى المزعومة",
        "none": "لا توجد شكاوى في هذه الفئة.",
        "brief": "الموجز",
        "brief_template": "هذا الموجز مصدره القالب الثابت، وليس نموذج الذكاء الاصطناعي.",
        "evidence": "أدلة من المراجعات",
        "low": "ثقة منخفضة",
        "columns": ("التقييم", "التاريخ", "الثقة", "نص المراجعة"),
        "checklist": "قائمة تحقق المفتش",
        "unset": "لم تتم المراجعة",
        "notes": "ملاحظات",
        "previous_title": "التفتيش السابق",
        "sample_title": "التفتيش السابق (بيانات نموذجية)",
        "sample_warning": "بيانات نموذجية للعرض فقط، وليست سجل تفتيش حقيقيا.",
        "sample_outcome": "سجل نموذجي، وليس تفتيشا حقيقيا",
        "generated": "تاريخ الإنشاء",
        "timezone": "بالتوقيت العالمي",
        "categories": {"safety": "السلامة", "hygiene": "النظافة", "licensing": "التراخيص",
                       "pricing_fraud": "الأسعار", "service_only": "الخدمة فقط"},
        "statuses": {"Confirmed": "مؤكد", "Ongoing": "قيد المتابعة", "False alarm": "إنذار خاطئ", "Resolved": "تم الحل"},
    },
}


def stars_text(value: float) -> str:
    """The stars come from the fallback font, because Plex does not include them."""
    filled = int(value + 0.5)
    return "★" * filled + "☆" * (5 - filled)


def complaint_line(language: str, hits: int, credible: int) -> str:
    if language == "ar":
        return f"عدد المراجعات {hits}، منها {credible} موثوقة"
    return f"{hits} {'review alleges' if hits == 1 else 'reviews allege'} this ({credible} credible)"


class Report(FPDF):
    def __init__(self, language: str):
        super().__init__(format="A4")
        self.language = language
        self.rtl = language == "ar"
        self.add_font("body", "", BRAND_FONTS / "IBMPlexSansArabic-Regular.ttf")
        self.add_font("body", "B", BRAND_FONTS / "IBMPlexSansArabic-SemiBold.ttf")
        # Plex lacks some letters quoted reviews use, such as Turkish and Cyrillic ones.
        self.add_font("fallback", "", FALLBACK_FONTS / "DejaVuSans.ttf")
        self.add_font("fallback", "B", FALLBACK_FONTS / "DejaVuSans-Bold.ttf")
        self.set_fallback_fonts(["fallback"], exact_match=False)
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(True, margin=18)
        self.add_page()

    def footer(self) -> None:
        self.set_y(-12)
        self.set_text_shaping(True, direction="ltr")
        self.set_font("body", "", 8)
        self.set_text_color(110)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(0)

    def printable(self, text: str) -> str:
        """Drop characters the font cannot draw, such as emoji, so no empty boxes appear."""
        body, fallback = self.fonts["body"].cmap, self.fonts["fallback"].cmap
        return "".join(c for c in text if c.isspace() or ord(c) in body or ord(c) in fallback)

    def paragraph(self, text: str, size: float = 10, bold: bool = False, own_direction: bool = False,
                  gap: float = 1.5, color: int | tuple[int, int, int] = 0) -> None:
        """own_direction keeps quoted reviews in their own language's direction."""
        text = self.printable(str(text))
        if own_direction:
            rtl = bool(ARABIC.search(text))
            direction = None
        else:
            rtl = self.rtl
            direction = "rtl" if rtl else "ltr"
        self.set_text_shaping(True, direction=direction)
        self.set_font("body", "B" if bold else "", size)
        self.set_text_color(*color) if isinstance(color, tuple) else self.set_text_color(color)
        height = size * (0.72 if rtl else 0.6)
        self.multi_cell(0, height, text, align="R" if rtl else "L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(gap)

    def heading(self, text: str) -> None:
        self.ln(4)
        self.paragraph(text, size=13, bold=True, color=MAROON, gap=0.6)
        self.set_draw_color(*RULE)
        self.set_line_width(0.25)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)


def translated_from(text: str, code: str) -> str:
    """Script beats the stored label, because labels on short reviews are unreliable."""
    if ARABIC.search(text):
        return "Translated from Arabic"
    name = LANGUAGE_NAMES.get(code)
    return f"Translated from {name}" if name else "Machine translated"


def translated_from_arabic_label(code: str) -> str:
    name = ARABIC_LANGUAGE_NAMES.get(code)
    return f"مترجم من {name}" if name else "ترجمة آلية"


def mostly_arabic(text: str) -> bool:
    arabic = len(ARABIC.findall(text))
    return arabic > 0 and arabic >= len(re.findall("[A-Za-z]", text))


def arabic_brief(brief: str) -> str:
    for latin, arabic in BRIEF_NAMES_AR.items():
        brief = brief.replace(latin, arabic)
    return brief


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def evidence_table(pdf: Report, text: dict, category: str, reviews: list[dict], translations: dict) -> None:
    # translations holds the report's own language, so every quote ends up in that language.
    """One table per category. Arabic reports run right to left, so the columns are reversed."""
    hits = [r for r in reviews if isinstance(r.get(f"{category}_evidence"), str) and not r[f"{category}_likely_praise"]]
    if not hits:
        return
    ranked = sorted(hits, key=lambda r: (r[f"{category}_low_confidence"], -r[f"{category}_confidence"]))[:MAX_EXAMPLES]
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()
    pdf.paragraph(text["categories"][category], size=10.5, bold=True, color=MAROON, gap=1)

    order = (lambda cells: list(reversed(cells))) if pdf.rtl else list
    side = Align.R if pdf.rtl else Align.L
    pdf.set_text_shaping(True, direction=None)
    pdf.set_font("body", "", 8.5)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.2)
    with pdf.table(
        col_widths=tuple(order(TABLE_WIDTHS)), text_align=side, v_align="TOP", line_height=5.2, padding=(1.8, 2),
        borders_layout="HORIZONTAL_LINES", headings_style=FontFace(emphasis="BOLD", color=(60, 60, 66), fill_color=TABLE_HEAD),
    ) as table:
        head = table.row()
        for label in order(list(text["columns"])):
            head.cell(label)
        for review in ranked:
            evidence = review[f"{category}_evidence"]
            translated = translations.get(evidence)
            shown = translated or " ".join(evidence.split())
            quote = shorten(pdf.printable(shown), MAX_TRANSLATION_CHARS if translated else MAX_EXAMPLE_CHARS)
            span = 2 if translated else 1
            confidence = f"{review[f'{category}_confidence']:.2f}"
            if review[f"{category}_low_confidence"]:
                confidence += f"\n{text['low']}"
            cells = [
                ("stars", stars_text(review["rating"])),
                ("plain", f"{review['timestamp_utc']:%Y-%m}"),
                ("plain", confidence),
                ("quote", quote),
            ]
            row = table.row()
            for kind, value in order(cells):
                if kind == "stars":
                    row.cell(value, style=FontFace(color=GOLD), rowspan=span)
                elif kind == "quote":
                    row.cell(value, align=Align.R if ARABIC.search(value) else Align.L)
                else:
                    row.cell(value, rowspan=span)
            if translated:
                label = translated_from_arabic_label(review["language"]) if pdf.rtl else translated_from(evidence, review["language"])
                table.row().cell(f"({label})", align=side, style=FontFace(color=TRANSLATION))
    pdf.ln(4)


def build_report(business: dict, reviews: list[dict], total_businesses: int, language: str,
                 checklist: dict, notes: str, sample: dict | None, translations: dict | None = None,
                 previous: dict | None = None) -> str:
    """translations maps original text to the report's language, including business names for Arabic reports."""
    translations = translations or {}
    """Writes the PDF and returns its path. Raises if the font is missing, rather than print broken text."""
    text = TEXT[language]
    pdf = Report(language)

    pdf.paragraph(text["title"], size=20, bold=True, gap=0.6, color=MAROON)
    pdf.paragraph(translations.get(business["business"], business["business"]), size=13.5, bold=True, gap=2)

    pdf.heading(text["summary"])
    for label, value in (
        (text["rank"], f"{business['rank']} {text['of']} {total_businesses}"),
        (text["risk"], f"{business['risk_score']:.2f}"),
        (text["rating"], f"{business['place_rating']} {stars_text(business['place_rating'])}"),
        (text["reviews"], f"{business['reviews_scored']}"),
        (text["credibility"], f"{business['mean_credibility']:.2f}"),
    ):
        pdf.paragraph(f"{label}: {value}", gap=0.4)
    pdf.ln(1)
    pdf.paragraph(text["sample_note"], size=8.5, color=90)

    pdf.heading(text["complaints"])
    for category in CATEGORY_ORDER:
        hits, credible = business[f"{category}_hits"], business[f"{category}_credible_hits"]
        line = complaint_line(language, hits, credible) if hits else text["none"]
        pdf.paragraph(f"{text['categories'][category]}: {line}", gap=0.4)

    pdf.heading(text["brief"])
    brief = business[f"brief_{language}"]
    pdf.paragraph(arabic_brief(brief) if language == "ar" else brief)
    if str(business[f"brief_source_{language}"]).startswith("template"):
        pdf.paragraph(text["brief_template"], size=8.5, color=90)

    pdf.heading(text["evidence"])
    for category in CATEGORY_ORDER:
        evidence_table(pdf, text, category, reviews, translations)

    pdf.heading(text["checklist"])
    for category in CATEGORY_ORDER:
        status = checklist.get(category)
        label = text["statuses"].get(status, text["unset"])
        pdf.paragraph(f"{text['categories'][category]}: {label}", gap=0.4)
    if notes and notes.strip():
        pdf.ln(1)
        pdf.paragraph(f"{text['notes']}:", bold=True, gap=0.4)
        pdf.paragraph(notes, own_direction=True)

    if previous:
        pdf.heading(text["previous_title"])
        pdf.paragraph(f"{previous['date']}: {previous['notes']}".rstrip(": "), own_direction=True)
    elif sample:
        pdf.heading(text["sample_title"])
        pdf.paragraph(text["sample_warning"], size=9, bold=True, color=80)
        pdf.paragraph(f"{sample['last_visit']}: {text.get('sample_outcome') or sample['outcome']}", own_direction=True)

    pdf.ln(3)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    pdf.paragraph(f"{text['generated']}: {stamp} {text['timezone']}", size=8, color=110)

    name = re.sub(r"[^a-z0-9]+", "_", business["business"].lower()).strip("_")
    path = Path(tempfile.mkdtemp()) / f"inspection_{name}_{language}.pdf"
    pdf.output(str(path))
    return str(path)
