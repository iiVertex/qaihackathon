# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.6", "protobuf==7.36.2", "pyarrow==26.0.0",
#   "sentencepiece==0.2.2", "torch==2.14.1", "transformers==5.20.2",
# ]
# ///
"""Phase 4 last step: % of 200 real English Qatar reviews flagged as AI-written."""
import sys
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "Hermesione/robert-fake-reviewer"
sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_parquet("data/scraped_reviews.parquet")
pool = df[(df["language"] == "en") & (df["sample"] == "newest")]
print("English newest pool:", len(pool))
rows = pool.sample(n=min(200, len(pool)), random_state=42).copy()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
print("id2label:", model.config.id2label)
scores = []
with torch.no_grad():
    for start in range(0, len(rows), 16):
        batch = tokenizer(rows["text"].iloc[start:start + 16].tolist(), padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        scores += torch.softmax(model(**batch).logits, dim=1)[:, 1].tolist()
rows["p_ai"] = scores
flagged = rows["p_ai"] >= 0.5
print(f"reviews: {len(rows)} | flagged AI: {flagged.sum()} ({flagged.mean():.1%}) | mean p_ai {rows['p_ai'].mean():.3f}")
rows["length"] = pd.cut(rows["text"].str.len(), [0, 50, 150, 100000], labels=["<=50 chars", "51-150", ">150"])
print(rows.groupby("length", observed=True)["p_ai"].agg(n="size", flagged=lambda s: (s >= 0.5).mean()).round(3).to_string())
print(rows.groupby("category")["p_ai"].agg(n="size", flagged=lambda s: (s >= 0.5).mean()).round(3).to_string())
print("\nhighest scoring:")
for row in rows.sort_values("p_ai", ascending=False).head(6).itertuples():
    print(f"  {row.p_ai:.3f} | {row.rating}* | {row.text[:110]!r}")
print("\nlowest scoring:")
for row in rows.sort_values("p_ai").head(3).itertuples():
    print(f"  {row.p_ai:.3f} | {row.rating}* | {row.text[:110]!r}")
