"""Every word the app shows, in English and Arabic. PDF wording lives in report.py."""

CHOICES = {"English": "en", "العربية": "ar"}
STATUSES = ("Confirmed", "Ongoing", "False alarm", "Resolved")

UI = {
    "en": {
        "title": "Review Inspector",
        "subtitle": "Ranked evidence from public reviews, for inspectors in Qatar.",
        "search": "Search for a business",
        "site_language": "Language",
        "open_account": "Sign in or create account",
        "sign_out": "Sign out",
        "email": "Email",
        "password": "Password",
        "sign_in": "Sign in",
        "create_account": "Create account",
        "enter_details": "Enter your email and password.",
        "confirm_email": "Check your email to confirm your account, then sign in.",
        "account_failed": "Something went wrong. Try again.",
        "mode_demo": "<strong>Demo mode.</strong> Try everything here, but nothing is stored. Create an account to save your inspections.",
        "mode_signed_in": "<strong>Signed in as {email}.</strong> Your checklists and notes are saved to your account.",
        "tab_complaints": "Complaints",
        "tab_brief": "Brief and report",
        "tab_checklist": "Checklist",
        "tab_inspections": "My inspections",
        "tab_businesses": "All businesses",
        "tab_method": "How this works",
        "intro_complaints": "Safety comes first. Highlighted text is the sentence that supports the complaint. Reviews in other languages are shown in English, with the original language in brackets.",
        "intro_checklist": "Record what you found on site. Your choices also appear in the PDF report.",
        "intro_inspections": "Restaurants with a saved checklist, newest first. Search a restaurant above to reopen it.",
        "report_language": "Report language",
        "create_pdf": "Create PDF report",
        "pdf_label": "PDF report",
        "notes": "Notes",
        "save": "Save checklist",
        "statuses": {"Confirmed": "Confirmed", "Ongoing": "Ongoing", "False alarm": "False alarm", "Resolved": "Resolved"},
        "alleged": "{label} ({count} alleged)",
        "rank_of": "{rank} of {total}",
        "priority_rank": "Priority rank",
        "priority_score": "Priority score",
        "google_rating": "Google rating",
        "reviews_analysed": "Reviews analysed",
        "average_credibility": "Average credibility",
        "place_line": "{kind} with {count} Google reviews",
        "kinds": {"hotel": "hotel", "restaurant": "restaurant"},
        "complaining": "What reviewers are complaining about",
        "concern_help": "Concern level compares this business with the other {others}. Read the reviews in the Complaints tab below.",
        "categories": {"safety": "Safety", "hygiene": "Hygiene", "licensing": "Licensing", "pricing_fraud": "Pricing",
                       "service_only": "Service only"},
        "category_help": {
            "safety": "Danger to people, such as fire hazards, injuries or exposed wiring",
            "hygiene": "Cleanliness, such as dirt, pests, bad smells or spoiled food",
            "licensing": "Possible rule breaking, such as alcohol, smoking or permits",
            "pricing_fraud": "Overcharging, hidden fees or refused refunds",
            "service_only": "Rude staff or slow service, with no rule broken",
        },
        "levels": {"High": "High concern", "Medium": "Medium concern", "Low": "Low concern", "None": "No complaints"},
        "one_review": "1 review",
        "many_reviews": "{count} reviews",
        "reliable": "{count} look reliable",
        "nothing_to_check": "Nothing to check",
        "confidence": "confidence {value:.2f}",
        "credibility": "credibility {value:.2f}",
        "likely_praise": "likely praise",
        "low_confidence": "low confidence",
        "whole_review": "whole review",
        "key_sentence": "key sentence",
        "not_shown": "{count} more not shown.",
        "praise_extra": ", plus {count} likely praise",
        "no_complaints": "No complaints found in the reviews analysed.",
        "written_by_ai": "Written by an AI model from the data above.",
        "written_by_template": "Written from a fixed template.",
        "sample_tag": "sample data",
        "previous_inspection": "Previous inspection: {date}, {visit}. {outcome}.",
        "visit_types": {},
        "outcomes": {},
        "demo_tag": "demo",
        "last_saved": "Last saved {date}, {saves}",
        "one_save": "1 save",
        "many_saves": "{count} saves",
        "history_empty_signed_in": "No inspections saved yet. Save a checklist and the restaurant appears here.",
        "history_empty_demo": "Nothing here yet. Save a checklist to see how this works, or create an account to keep your inspections.",
        "demo_kept": "Demo only: this is kept until you leave the page.",
        "restored": "Last saved on {date}. Your earlier answers are filled in below.",
        "demo_saved": "Demo only: kept until you leave the page. Create an account to save it for good.",
        "saved": "Saved for {name}. It is now in My inspections.",
        "need_status": "Choose at least one status or add a note first.",
        "notes_too_long": "Notes can be up to {limit} characters.",
        "stars_label": "{value:g} out of 5 stars",
        "table": ("Rank", "Business", "Type", "Priority", "Google rating", "Reviews analysed"),
        "method": """
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
""",
    },
    "ar": {
        "title": "مفتش المراجعات",
        "subtitle": "أدلة مرتبة من المراجعات العامة لمساعدة المفتشين في دولة قطر.",
        "search": "ابحث عن منشأة",
        "site_language": "اللغة",
        "open_account": "تسجيل الدخول أو إنشاء حساب",
        "sign_out": "تسجيل الخروج",
        "email": "البريد الإلكتروني",
        "password": "كلمة المرور",
        "sign_in": "تسجيل الدخول",
        "create_account": "إنشاء حساب",
        "enter_details": "أدخل بريدك الإلكتروني وكلمة المرور.",
        "confirm_email": "تحقق من بريدك الإلكتروني لتأكيد حسابك، ثم سجل الدخول.",
        "account_failed": "حدث خطأ ما. حاول مرة أخرى.",
        "mode_demo": "<strong>الوضع التجريبي.</strong> جرّب كل شيء هنا، لكن لن يُحفظ أي شيء. أنشئ حسابا لحفظ عمليات التفتيش.",
        "mode_signed_in": "<strong>تم تسجيل الدخول باسم {email}.</strong> تُحفظ قوائم التحقق والملاحظات في حسابك.",
        "tab_complaints": "الشكاوى",
        "tab_brief": "الموجز والتقرير",
        "tab_checklist": "قائمة التحقق",
        "tab_inspections": "عمليات التفتيش الخاصة بي",
        "tab_businesses": "جميع المنشآت",
        "tab_method": "طريقة العمل",
        "intro_complaints": "تأتي السلامة أولا. النص المظلل هو الجملة التي تدعم الشكوى. تُعرض المراجعات المكتوبة بلغات أخرى باللغة العربية، مع ذكر اللغة الأصلية بين قوسين.",
        "intro_checklist": "سجّل ما وجدته في الموقع. تظهر اختياراتك أيضا في تقرير PDF.",
        "intro_inspections": "المنشآت التي لها قائمة تحقق محفوظة، والأحدث أولا. ابحث عن المنشأة في الأعلى لفتحها مجددا.",
        "report_language": "لغة التقرير",
        "create_pdf": "إنشاء تقرير PDF",
        "pdf_label": "تقرير PDF",
        "notes": "ملاحظات",
        "save": "حفظ قائمة التحقق",
        "statuses": {"Confirmed": "مؤكد", "Ongoing": "قيد المتابعة", "False alarm": "إنذار خاطئ", "Resolved": "تم الحل"},
        "alleged": "{label} ({count} شكوى)",
        "rank_of": "{rank} من {total}",
        "priority_rank": "ترتيب الأولوية",
        "priority_score": "درجة الأولوية",
        "google_rating": "تقييم جوجل",
        "reviews_analysed": "المراجعات المحللة",
        "average_credibility": "متوسط المصداقية",
        "place_line": "{kind}، عدد المراجعات على جوجل: {count}",
        "kinds": {"hotel": "فندق", "restaurant": "مطعم"},
        "complaining": "ما الذي يشتكي منه المراجعون",
        "concern_help": "يقارن مستوى القلق هذه المنشأة بالمنشآت التسع الأخرى. اقرأ المراجعات في تبويب الشكاوى أدناه.",
        "categories": {"safety": "السلامة", "hygiene": "النظافة", "licensing": "التراخيص", "pricing_fraud": "الأسعار",
                       "service_only": "الخدمة فقط"},
        "category_help": {
            "safety": "خطر على الأشخاص، مثل مخاطر الحريق أو الإصابات أو الأسلاك المكشوفة",
            "hygiene": "النظافة، مثل الأوساخ أو الحشرات أو الروائح الكريهة أو الطعام الفاسد",
            "licensing": "مخالفات محتملة، مثل الكحول أو التدخين أو التصاريح",
            "pricing_fraud": "المبالغة في الأسعار أو الرسوم الخفية أو رفض استرداد المبالغ",
            "service_only": "سوء تعامل الموظفين أو بطء الخدمة، دون مخالفة أي نظام",
        },
        "levels": {"High": "قلق مرتفع", "Medium": "قلق متوسط", "Low": "قلق منخفض", "None": "لا توجد شكاوى"},
        "one_review": "مراجعة واحدة",
        "many_reviews": "عدد المراجعات: {count}",
        "reliable": "الموثوقة منها: {count}",
        "nothing_to_check": "لا يوجد ما يستدعي التحقق",
        "confidence": "الثقة {value:.2f}",
        "credibility": "المصداقية {value:.2f}",
        "likely_praise": "ثناء محتمل",
        "low_confidence": "ثقة منخفضة",
        "whole_review": "المراجعة كاملة",
        "key_sentence": "الجملة الأساسية",
        "not_shown": "لم تُعرض {count} مراجعة أخرى.",
        "praise_extra": "، إضافة إلى {count} ثناء محتمل",
        "no_complaints": "لم يُعثر على شكاوى في المراجعات المحللة.",
        "written_by_ai": "كتبه نموذج ذكاء اصطناعي استنادا إلى البيانات أعلاه.",
        "written_by_template": "كُتب من قالب ثابت.",
        "sample_tag": "بيانات نموذجية",
        "previous_inspection": "التفتيش السابق: {date}، {visit}. {outcome}.",
        "visit_types": {"Routine": "زيارة روتينية"},
        "outcomes": {"Sample record, not a real inspection": "سجل نموذجي، وليس تفتيشا حقيقيا"},
        "demo_tag": "تجريبي",
        "last_saved": "آخر حفظ {date}، {saves}",
        "one_save": "حفظ واحد",
        "many_saves": "عدد مرات الحفظ: {count}",
        "history_empty_signed_in": "لا توجد عمليات تفتيش محفوظة بعد. احفظ قائمة تحقق وستظهر المنشأة هنا.",
        "history_empty_demo": "لا يوجد شيء هنا بعد. احفظ قائمة تحقق لترى كيف يعمل ذلك، أو أنشئ حسابا للاحتفاظ بعمليات التفتيش.",
        "demo_kept": "تجريبي فقط: يبقى هذا حتى تغادر الصفحة.",
        "restored": "آخر حفظ في {date}. إجاباتك السابقة معبأة أدناه.",
        "demo_saved": "تجريبي فقط: يبقى حتى تغادر الصفحة. أنشئ حسابا لحفظه بشكل دائم.",
        "saved": "تم الحفظ لـ {name}. أصبح الآن في عمليات التفتيش الخاصة بي.",
        "need_status": "اختر حالة واحدة على الأقل أو أضف ملاحظة أولا.",
        "notes_too_long": "يمكن أن تصل الملاحظات إلى {limit} حرف.",
        "stars_label": "{value:g} من 5 نجوم",
        "table": ("الترتيب", "المنشأة", "النوع", "الأولوية", "تقييم جوجل", "المراجعات المحللة"),
        "method": """
- **المراجعات:** ما يصل إلى 50 من أحدث المراجعات المكتوبة و50 من أقلها تقييما لكل منشأة. ولأن التقييمات المنخفضة ممثلة بكثرة في العينة، فإن الأعداد ليست نسبا من جميع المراجعات.
- **درجة الأولوية:** تجمع شدة الشكاوى في كل فئة. تعني الدرجة الأعلى أن تبدأ بهذه المنشأة. وهي مقارنة بين هذه المنشآت، وليست تقييما للسلامة.
- **مستويات القلق:** تقارن كل فئة المنشأة بالمنشآت التسع الأخرى. القلق المرتفع يعني أعلى ثلاث منشآت، والمتوسط يعني الثلاث التالية، والمنخفض يعني البقية. هذه مقارنات، وليست تقييمات للسلامة.
- **الموثوقة:** مراجعة اجتازت فحوص المصداقية وتتضمن جملة تدعم الشكوى.
- **المخالفات:** سؤال بنعم أو لا من نموذج GLiNER2 لكل فئة. لم تُقس النتائج على بيانات مصنفة، لذا توقع إنذارات خاطئة وحالات لم تُكتشف.
- **فحص النص المكتوب بالذكاء الاصطناعي:** للمراجعات الإنجليزية فقط. أخطأ في تصنيف نحو واحدة من كل 7 مراجعات بشرية عند اختباره على مراجعات فنادق، لذا يخفض المصداقية قليلا فقط.
- **المصداقية والشدة:** أوزان محددة يدويا وليست معايرة. الشدة = وزن الفئة × الثقة × الشكاوى الموثوقة لكل 1000 مراجعة على جوجل × حداثة الشكوى.
- **الترتيب العادل:** تُحتسب الشكاوى وفق توزيع النجوم في جوجل، حتى لا تُرتب المنشأة حسب حجمها فقط. وهذا تقدير.
- **النظافة والسلامة:** لا تُحذف هذه النتائج أبدا، بل تُعلّم بأنها منخفضة الثقة. وتُعلّم النتائج في مراجعات 4 و5 نجوم والمراجعات الإيجابية بثلاث نجوم بأنها ثناء محتمل ولا تضاف إلى الشدة. وكذلك النتائج التي لا تدعمها جملة واحدة.
- **الترجمة:** تُعرض المراجعات المكتوبة بلغات أخرى باللغة العربية، بترجمة من نموذج ذكاء اصطناعي (GLM-5.3)، مع ذكر اللغة الأصلية بين قوسين. وبالنسبة إلى هذه المراجعات، تعرض الصفحة الجملة الأساسية من المراجعة.
- **الوضع التجريبي والحسابات:** يمكن لأي شخص تجربة التطبيق. ولا يحفظ عمله إلا المفتشون الذين لديهم حساب، ويرى كل مفتش عمله فقط.
- **نقطة ضعف معروفة:** نتائج السلامة والتراخيص هي الأقل دقة، لذا تحقق من أدلتها قبل اتخاذ أي إجراء.
""",
    },
}

# Supabase answers in English, so its common messages are translated here.
ACCOUNT_ERRORS_AR = {
    "Invalid login credentials": "البريد الإلكتروني أو كلمة المرور غير صحيحة.",
    "User already registered": "هذا البريد الإلكتروني مسجل بالفعل. سجل الدخول بدلا من ذلك.",
    "Email not confirmed": "لم يتم تأكيد بريدك الإلكتروني بعد.",
    "Password should be at least 6 characters.": "يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.",
    "Password should be at least 6 characters": "يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.",
    "Could not reach the account service. Check your connection and try again.": "تعذر الوصول إلى خدمة الحسابات. تحقق من اتصالك وحاول مرة أخرى.",
}


def account_error(code: str, message: str) -> str:
    if code == "en":
        return message
    return ACCOUNT_ERRORS_AR.get(message, UI["ar"]["account_failed"])
