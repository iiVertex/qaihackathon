# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "gliner2[local]==2.1.0",
#   "numpy==2.4.6",
#   "pandas==3.0.6",
#   "protobuf==7.36.2",
#   "pyarrow==26.0.0",
#   "sentencepiece==0.2.2",
#   "torch==2.14.1",
#   "transformers==5.20.2",
# ]
# ///
"""Phase 3: run GLiNER2 on 30 Arabic + 30 English real reviews, for human judgement."""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from gliner2 import AutoExtractor

MODEL = "fastino/GLiNER2.5-multi-Decide"
THRESHOLD = 0.4
SEED = 42
MIN_CHARS = 30
OUTPUT = Path("data/gliner_check.json")

TASKS = {
    "violations": {
        "labels": {
            "hygiene": "Food poisoning, pests, dirty kitchen or rooms, spoiled food, bad smells",
            "safety": "Fire hazards, blocked exits, injuries, broken structures, electrical danger",
            "pricing_fraud": "Overcharging, hidden fees, bill different from menu, refused refund",
            "licensing": "Unlicensed activity, alcohol or smoking violations, expired permits",
            "service_only": "Rude staff or slow service, with no regulatory issue",
            "no_complaint": "A positive review, or one with no complaint at all",
        },
        "multi_label": True,
        "cls_threshold": THRESHOLD,
    }
}


def pick(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """20 low-rated and 10 high-rated reviews, so both hits and non-hits get tested."""
    rows = df[(df["language"] == language) & (df["text"].str.len() >= MIN_CHARS)]
    low = rows[rows["rating"] <= 2].sample(n=20, random_state=SEED)
    high = rows[rows["rating"] >= 4].sample(n=10, random_state=SEED)
    return pd.concat([low, high])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_parquet("data/scraped_reviews.parquet")
    chosen = pd.concat([pick(df, "en"), pick(df, "ar")])
    print(f"reviews: {len(chosen)} | {chosen['language'].value_counts().to_dict()}")

    started = time.time()
    model = AutoExtractor.from_pretrained(MODEL)
    print(f"model loaded in {time.time() - started:.0f}s")

    results = []
    started = time.time()
    for row in chosen.itertuples():
        output = model.classify_text(row.text, TASKS, include_confidence=True, include_spans=True)
        results.append({"language": row.language, "rating": row.rating, "business": row.business,
                        "sample": row.sample, "text": row.text, "output": output})
    seconds = time.time() - started
    print(f"classified {len(results)} in {seconds:.0f}s ({seconds / len(results):.2f}s per review)")

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"saved {OUTPUT}")
    print("first output:", json.dumps(results[0]["output"], ensure_ascii=False, default=str)[:600])


if __name__ == "__main__":
    main()
