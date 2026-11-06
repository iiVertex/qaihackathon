# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "accelerate==1.15.0",
#   "huggingface_hub==2.4.0",
#   "numpy==2.4.6",
#   "pandas==3.0.6",
#   "protobuf==7.36.2",
#   "pyarrow==26.0.0",
#   "scikit-learn==1.9.1",
#   "sentencepiece==0.2.2",
#   "torch==2.14.1",
#   "transformers==5.20.2",
# ]
# ///
"""Phase 4: train the AI-text classifier (xlm-roberta-base) on MAiDE-up.

MAiDE-up ships a single split, so the held-out sets are built here by hashing the
hotel name: 10% of hotels are test, 10% validation, 80% train. A hotel never
crosses splits, and the same hotels are held out whichever languages are trained.

  --smoke               ~200 rows, 1 epoch, pushes metrics to smoke/metrics.json only
  --languages English   default. Pass "all" for the 10 languages.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "FacebookAI/xlm-roberta-base"
DATASET_REPO = "MichiganNLP/MAiDE-up"
MODEL_REPO = "Hermesione/robert-fake-reviewer"
SEED = 42
SMOKE_ROWS = 200
LABELS = {0: "human", 1: "ai"}


def split_of(hotel: str) -> str:
    bucket = int(hashlib.sha256(hotel.encode("utf-8")).hexdigest(), 16) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def load_data(languages: str, smoke: bool) -> pd.DataFrame:
    path = hf_hub_download(
        DATASET_REPO, "default/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"
    )
    df = pd.read_parquet(path)
    print(f"loaded {len(df)} rows from {DATASET_REPO}")

    df["text"] = (df["Upside_Review"].fillna("") + " " + df["Downside_Review"].fillna("")).str.strip()
    df["split"] = df["Hotel Name"].map(split_of)
    df["label"] = df["source"].astype(int)  # 1 = generated from a prompt, 0 = real Booking review

    df = df[df["text"].str.len() > 0]
    before = len(df)
    # Identical text under different hotels would leak test rows into training.
    df = df.drop_duplicates(subset="text", keep=False)
    print(f"dropped {before - len(df)} rows with empty or duplicated text")

    if languages != "all":
        df = df[df["Review_Language"].isin(languages.split(","))]
    if smoke:
        df = df.sample(n=min(SMOKE_ROWS, len(df)), random_state=SEED)
    print("rows by split and label:\n", pd.crosstab(df["split"], df["label"]).to_string())
    print("hotels by split:", df.groupby("split")["Hotel Name"].nunique().to_dict())
    return df


class Reviews(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int):
        self.encodings = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_len)
        self.labels = frame["label"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def scores(labels, predictions) -> dict:
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def compute_metrics(prediction) -> dict:
    return scores(prediction.label_ids, np.argmax(prediction.predictions, axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--languages", default="English", help='comma-separated, or "all"')
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("ERROR: HF_TOKEN is not set, so results could not be pushed. Stopping before training.")
    gpu = torch.cuda.is_available()
    print(f"gpu: {torch.cuda.get_device_name(0) if gpu else 'none'} | args: {vars(args)}")

    frame = load_data(args.languages, args.smoke)
    parts = {name: frame[frame["split"] == name] for name in ("train", "validation", "test")}
    empty = [name for name, part in parts.items() if part.empty]
    if empty:
        sys.exit(f"ERROR: empty splits: {empty}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}
    )
    epochs = 1 if args.smoke else args.epochs
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            fp16=gpu,
            seed=SEED,
            report_to="none",
        ),
        train_dataset=Reviews(parts["train"], tokenizer, args.max_len),
        eval_dataset=Reviews(parts["validation"], tokenizer, args.max_len),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # The test split is used only here, after training has finished.
    test = parts["test"]
    logits = trainer.predict(Reviews(test, tokenizer, args.max_len)).predictions
    predicted = np.argmax(logits, axis=1)
    labels = test["label"].to_numpy()
    metrics = {
        "split": "held-out test: 10% of hotels by SHA-256 of hotel name, never used in training",
        "smoke": args.smoke,
        "base_model": BASE_MODEL,
        "dataset": DATASET_REPO,
        "languages_trained": args.languages,
        "epochs": epochs,
        "max_len": args.max_len,
        "rows": {name: int(len(part)) for name, part in parts.items()},
        "test_overall": scores(labels, predicted),
        "test_by_language": {
            language: scores(labels[mask], predicted[mask])
            for language in sorted(test["Review_Language"].unique())
            for mask in [(test["Review_Language"] == language).to_numpy()]
        },
    }
    print(json.dumps(metrics, indent=2))

    output = Path("model_output")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("saved:", sorted(p.name for p in output.iterdir()))

    hub = HfApi()
    if args.smoke:
        hub.upload_file(path_or_fileobj=str(output / "metrics.json"), path_in_repo="smoke/metrics.json",
                        repo_id=MODEL_REPO, commit_message="Smoke test metrics")
    else:
        hub.upload_folder(folder_path=str(output), repo_id=MODEL_REPO,
                          commit_message=f"Train on MAiDE-up ({args.languages})")
    print(f"pushed: https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
