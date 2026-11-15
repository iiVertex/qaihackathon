# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "gliner2[local]==2.1.0",
#   "numpy==2.4.6",
#   "protobuf==7.36.2",
#   "sentencepiece==0.2.2",
#   "torch==2.14.1",
#   "transformers==5.20.2",
# ]
# ///
"""Phase 3 rerun: one yes/no question per category on the same 60 reviews."""
import json
import re
import sys
import time
from pathlib import Path

from gliner2 import AutoExtractor

MODEL = "fastino/GLiNER2.5-multi-Decide"
QUESTIONS = {
    "hygiene": "Does the reviewer report a hygiene problem: food poisoning, pests, a dirty kitchen or room, spoiled food, or a bad smell?",
    "safety": "Does the reviewer report a safety danger: a fire hazard, blocked exit, injury, broken structure, or electrical problem?",
    "pricing_fraud": "Does the reviewer report being overcharged, hidden fees, a bill different from the menu, or a refused refund?",
    "licensing": "Does the reviewer report unlicensed activity, an alcohol or smoking violation, or an expired permit?",
    "service_only": "Is the reviewer complaining only about rude staff or slow service?",
}
TASKS = {name: {"labels": ["yes", "no"], "prompt": prompt} for name, prompt in QUESTIONS.items()}
SENTENCES = re.compile(r"(?<=[.!?؟])\s+|\s*[،؛]\s*")


def parse(value) -> tuple[str, float | None]:
    if isinstance(value, dict):
        return value.get("label"), value.get("confidence")
    if isinstance(value, list) and value:
        return parse(value[0])
    return value, None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    source = json.loads(Path("data/gliner_check.json").read_text(encoding="utf-8"))
    model = AutoExtractor.from_pretrained(MODEL)
    results, started = [], time.time()
    for row in source:
        text = " ".join(re.sub(r"<br\s*/?>", " ", row["text"]).split())
        raw = model.classify_text(text, TASKS, include_confidence=True)
        answers = {name: parse(raw.get(name)) for name in QUESTIONS}
        evidence = {}
        for name, (label, _) in answers.items():
            if label != "yes":
                continue
            best, best_score = text, -1.0
            for sentence in (s for s in SENTENCES.split(text) if len(s) >= 8):
                answer, score = parse(model.classify_text(sentence, {name: TASKS[name]}, include_confidence=True).get(name))
                score = (score or 0) if answer == "yes" else 0
                if score > best_score:
                    best, best_score = sentence, score
            evidence[name] = best
        results.append({"language": row["language"], "rating": row["rating"], "text": text,
                        "old": row["output"], "answers": answers, "evidence": evidence, "raw": raw})
    seconds = time.time() - started
    print(f"classified {len(results)} in {seconds:.0f}s ({seconds / len(results):.2f}s per review)")
    Path("data/gliner_yesno.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("first raw:", json.dumps(results[0]["raw"], ensure_ascii=False, default=str)[:500])


if __name__ == "__main__":
    main()
