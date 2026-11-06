"""PDF inspection report for one business, in English or Arabic."""
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

BRAND_FONTS = Path(__file__).parent / "fonts"
FALLBACK_FONTS = Path(os.environ.get("REPORT_FONT_DIR", "/usr/share/fonts/truetype/dejavu"))
MAROON = (138, 21, 56)
GOLD = (222, 146, 0)
RULE = (214, 214, 219)
CATEGORY_ORDER = ("safety", "hygiene", "licensing", "pricing_fraud", "service_only")
MAX_EXAMPLES = 3
MAX_EXAMPLE_CHARS = 280
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

TEXT = {
    "en": {
        "title": "Review Inspector report",
        "banner": "Unverified allegations from public Google Maps reviews. These are not inspection findings. "
                  "Credibility is an estimate, never a verdict that a review is fake.",
        "summary": "Summary",
        "rank": "Risk rank",
        "of": "of",
        "risk": "Risk score",
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
        "checklist": "Inspector checklist",
        "unset": "Not reviewed",
        "notes": "Notes",
        "sample_title": "Previous inspection (sample data)",
        "sample_warning": "Sample data for demonstration only. This is not a real inspection record.",
        "generated": "Generated",
        "categories": {"safety": "Safety", "hygiene": "Hygiene", "licensing": "Licensing",
                       "pricing_fraud": "Pricing", "service_only": "Service only"},
        "statuses": {"Confirmed": "Confirmed", "False alarm": "False alarm", "Resolved": "Resolved"},
    },
    "ar": {
        "title": "تقرير مفتش المراجعات",
        "banner": "ادعاءات غير مؤكدة مصدرها مراجعات عامة على خرائط Google، وليست نتائج تفتيش. "
                  "درجة المصداقية تقدير فقط، ولا تعني أن المراجعة مزيفة.",
        "summary": "الملخص",
        "rank": "ترتيب المخاطر",
        "of": "من",
        "risk": "درجة المخاطر",
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
        "checklist": "قائمة تحقق المفتش",
        "unset": "لم تتم المراجعة",
        "notes": "ملاحظات",
        "sample_title": "التفتيش السابق (بيانات نموذجية)",
        "sample_warning": "بيانات نموذجية للعرض فقط، وليست سجل تفتيش حقيقيا.",
        "generated": "تاريخ الإنشاء",
        "categories": {"safety": "السلامة", "hygiene": "النظافة", "licensing": "التراخيص",
                       "pricing_fraud": "الأسعار", "service_only": "الخدمة فقط"},
        "statuses": {"Confirmed": "مؤكد", "False alarm": "إنذار خاطئ", "Resolved": "تم الحل"},
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


def build_report(business: dict, reviews: list[dict], total_businesses: int, language: str,
                 checklist: dict, notes: str, sample: dict | None) -> str:
    """Writes the PDF and returns its path. Raises if the font is missing, rather than print broken text."""
    text = TEXT[language]
    pdf = Report(language)

    pdf.paragraph(text["title"], size=20, bold=True, gap=0.6, color=MAROON)
    pdf.paragraph(business["business"], size=13.5, bold=True, gap=2)
    pdf.paragraph(text["banner"], size=9, color=80)

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
    pdf.paragraph(business[f"brief_{language}"])
    if str(business[f"brief_source_{language}"]).startswith("template"):
        pdf.paragraph(text["brief_template"], size=8.5, color=90)

    pdf.heading(text["evidence"])
    for category in CATEGORY_ORDER[:-1]:
        hits = [r for r in reviews if isinstance(r.get(f"{category}_evidence"), str) and not r[f"{category}_likely_praise"]]
        if not hits:
            continue
        pdf.paragraph(text["categories"][category], bold=True, gap=0.6)
        ranked = sorted(hits, key=lambda r: (r[f"{category}_low_confidence"], -r[f"{category}_confidence"]))
        for review in ranked[:MAX_EXAMPLES]:
            quote = review[f"{category}_evidence"][:MAX_EXAMPLE_CHARS]
            pdf.paragraph(stars_text(review["rating"]), size=10, color=GOLD, gap=0.2)
            if review[f"{category}_low_confidence"]:
                pdf.paragraph(text["low"], size=8.5, color=90, gap=0.2)
            pdf.paragraph(quote, size=9.5, own_direction=True, gap=2)

    pdf.heading(text["checklist"])
    for category in CATEGORY_ORDER:
        status = checklist.get(category)
        label = text["statuses"].get(status, text["unset"])
        pdf.paragraph(f"{text['categories'][category]}: {label}", gap=0.4)
    if notes and notes.strip():
        pdf.ln(1)
        pdf.paragraph(f"{text['notes']}:", bold=True, gap=0.4)
        pdf.paragraph(notes, own_direction=True)

    if sample:
        pdf.heading(text["sample_title"])
        pdf.paragraph(text["sample_warning"], size=9, bold=True, color=80)
        pdf.paragraph(f"{sample['last_visit']}: {sample['outcome']}", own_direction=True)

    pdf.ln(3)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.paragraph(f"{text['generated']}: {stamp}", size=8, color=110)

    name = re.sub(r"[^a-z0-9]+", "_", business["business"].lower()).strip("_")
    path = Path(tempfile.mkdtemp()) / f"inspection_{name}_{language}.pdf"
    pdf.output(str(path))
    return str(path)
