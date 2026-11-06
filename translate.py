# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface_hub==2.4.0",
#   "numpy==2.4.6",
#   "pandas==3.0.6",
#   "pyarrow==26.0.0",
# ]
# ///
"""Translate non-English reviews into English once, and save them for the app.

Translations live in their own file, so rerunning pipeline.py never pays for them again.
  --dry-run   count what would be translated, with no API calls
  --limit N   test run on N texts, pushed to smoke/
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, InferenceClient, hf_hub_download

DATASET_REPO = "Hermesione/reviewere-dataset"
MODEL = "zai-org/GLM-5.3"
OUTPUT = Path("translations.parquet")
WORKERS = 4
MAX_TOKENS = 6000
MAX_CALLS = 420  # Hard stop: 387 approved texts plus a few retries.
MIN_LATIN_WORDS = 4  # Labels on short Latin text are unreliable: "Good" was tagged Somali.
SEED = 42
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
NON_LATIN = re.compile(r"[^\W\d_A-Za-zÀ-ɏ]")
SYSTEM = (
    "You translate customer reviews into natural English. Reviews may be in Gulf or other dialects of Arabic, or other "
    "languages, with slang, typos and transliterated English such as بوكينق for Booking. Keep the reviewer's meaning, tone "
    "and every detail, including names and numbers. Never add, soften or guess at facts. If a part is unclear, translate "
    "it as literally as you can. Output only the translation."
)


def needs_translation(text: str, language: str) -> bool:
    letters = len(LATIN.findall(text)) + len(NON_LATIN.findall(text))
    if letters == 0:
        return False
    if len(NON_LATIN.findall(text)) / letters > 0.5:
        return True
    return language != "en" and len(text.split()) >= MIN_LATIN_WORDS


def collect_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Whole reviews, plus evidence sentences, because the PDF quotes those on their own."""
    rows = {}
    evidence = [c for c in df.columns if c.endswith("_evidence")]
    for row in df.to_dict("records"):
        if not needs_translation(row["text"], row["language"]):
            continue
        rows[row["text"]] = "review"
        for column in evidence:
            value = row[column]
            if isinstance(value, str) and value not in rows and needs_translation(value, row["language"]):
                rows[value] = "evidence"
    return pd.DataFrame({"text": list(rows), "kind": list(rows.values())})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(hf_hub_download(DATASET_REPO, "reviews.parquet", repo_type="dataset"))
    texts = collect_texts(df)
    skipped = df[(df["language"] != "en") & ~df["text"].isin(texts["text"])]
    print(f"reviews {len(df)} | to translate {len(texts)} {texts['kind'].value_counts().to_dict()} | "
          f"non-English labels skipped {len(skipped)}: {skipped['text'].str[:18].unique()[:8].tolist()}")
    if args.dry_run:
        return
    if args.limit:
        texts = texts.sample(n=args.limit, random_state=SEED)
    if len(texts) > MAX_CALLS:
        sys.exit(f"ERROR: {len(texts)} texts is over the approved cap of {MAX_CALLS}. No API calls were made.")

    client = InferenceClient(provider="auto")
    calls, tokens = [0], [0, 0]

    def translate(text: str) -> str | None:
        for _ in range(2):  # Thinking can use up the token budget and return nothing, so allow one retry.
            if calls[0] >= MAX_CALLS:
                return None
            calls[0] += 1
            response = client.chat_completion(model=MODEL, max_tokens=MAX_TOKENS, temperature=0.1,
                                              messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
            tokens[0] += response.usage.prompt_tokens
            tokens[1] += response.usage.completion_tokens
            result = " ".join((response.choices[0].message.content or "").split())
            if result:
                return result
        return None

    def safe(text: str) -> str | None:
        try:
            return translate(text)
        except Exception as error:  # A failure is recorded and reported, never replaced with made-up text.
            print(f"FAILED {type(error).__name__}: {str(error)[:150]}", flush=True)
            return None

    started = time.time()
    results = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for index, result in enumerate(pool.map(safe, texts["text"]), 1):
            results.append(result)
            if index % 25 == 0 or index == len(texts):
                print(f"translated {index}/{len(texts)} in {time.time() - started:.0f}s", flush=True)
    texts = texts.assign(text_en=results)
    failed = int(texts["text_en"].isna().sum())
    cost = tokens[0] / 1e6 * 1.4 + tokens[1] / 1e6 * 4.4
    print(f"calls {calls[0]} | tokens in {tokens[0]} out {tokens[1]} | about ${cost:.2f} at $1.40/$4.40 | failed {failed}")

    texts.to_parquet(OUTPUT, index=False)
    path = f"smoke/{OUTPUT.name}" if args.limit else OUTPUT.name
    HfApi().upload_file(path_or_fileobj=str(OUTPUT), path_in_repo=path, repo_id=DATASET_REPO, repo_type="dataset",
                        commit_message=f"English translations ({len(texts)} texts)")
    print(f"pushed: https://huggingface.co/datasets/{DATASET_REPO}/blob/main/{path}")
    if args.limit:
        for row in texts.itertuples():
            print(f"\n[{row.kind}] {row.text[:110]}\n  => {str(row.text_en)[:170]}")


if __name__ == "__main__":
    main()
