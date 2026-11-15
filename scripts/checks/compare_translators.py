# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "accelerate==1.15.0",
#   "huggingface_hub==0.36.0",
#   "numpy==2.4.6",
#   "pandas==3.0.6",
#   "protobuf==7.36.2",
#   "pyarrow==26.0.0",
#   "sentencepiece==0.2.2",
#   "torch==2.14.1",
#   "transformers==4.56.0",
# ]
# ///
"""Runs translation models on the same real reviews, so a person can compare them.

NLLB was tested first and invented details, so this run covers the other two.
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

DATASET_REPO = "Hermesione/reviewere-dataset"
OUTPUT = Path("translation_compare2.json")
SEED = 42
ARABIC_ROWS = 24
OTHER_ROWS = 6
MAX_NEW_TOKENS = 256
ARABIC = re.compile(r"[؀-ۿ]")
NLLB_CODES = {"ar": "arb_Arab", "fr": "fra_Latn", "tr": "tur_Latn", "ru": "rus_Cyrl", "pt": "por_Latn", "es": "spa_Latn",
              "ko": "kor_Hang", "uk": "ukr_Cyrl", "pl": "pol_Latn", "nl": "nld_Latn", "de": "deu_Latn", "it": "ita_Latn"}


def pick_reviews() -> pd.DataFrame:
    path = hf_hub_download(DATASET_REPO, "reviews.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    arabic = df[df["text"].str.contains(ARABIC) & df["text"].str.len().between(25, 400)]
    # Complaints matter most, and they are where dialect and slang are hardest.
    low = arabic[arabic["rating"] <= 2].sample(n=ARABIC_ROWS - 6, random_state=SEED)
    high = arabic[arabic["rating"] >= 4].sample(n=6, random_state=SEED)
    other = df[df["language"].isin(NLLB_CODES) & (df["language"] != "ar") & (df["text"].str.len() >= 40)]
    other = other.sample(n=min(OTHER_ROWS, len(other)), random_state=SEED)
    return pd.concat([low, high, other])[["language", "rating", "text"]].reset_index(drop=True)


def hunyuan(rows: pd.DataFrame) -> list[str]:
    name = "tencent/Hunyuan-MT-7B"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    print("hunyuan loaded", flush=True)
    results = []
    for text in rows["text"]:
        messages = [{"role": "user", "content": f"Translate the following segment into English, without additional explanation.\n\n{text}"}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, disable_compile=True)
        results.append(tokenizer.decode(output[0][tokens.shape[1]:], skip_special_tokens=True).strip())
    return results


def seq2seq(name: str, rows: pd.DataFrame, forced_bos: str | None, prefix: str = "") -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    print(f"{name} loaded", flush=True)
    results = []
    for row in rows.itertuples():
        kwargs = {}
        if forced_bos:
            tokenizer.src_lang = NLLB_CODES.get(row.language, "arb_Arab")
            kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(forced_bos)
        tokens = tokenizer(prefix + row.text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokens, max_new_tokens=MAX_NEW_TOKENS, disable_compile=True, **kwargs)
        results.append(tokenizer.decode(output[0], skip_special_tokens=True).strip())
    return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    rows = pick_reviews()
    print(f"reviews: {len(rows)} | {rows['language'].value_counts().to_dict()}", flush=True)
    runs = {
        "hunyuan": lambda: hunyuan(rows),
        "madlad": lambda: seq2seq("google/madlad400-3b-mt", rows, None, "<2en> "),
    }
    timings = {}
    for name, run in runs.items():
        started = time.time()
        try:
            rows[name] = run()
            timings[name] = round((time.time() - started) / len(rows), 2)
        except Exception as error:  # One failed model must not lose the other two.
            rows[name] = f"FAILED: {type(error).__name__}: {str(error)[:120]}"
            timings[name] = None
            print(f"{name} FAILED: {type(error).__name__}: {str(error)[:300]}", flush=True)
        torch.cuda.empty_cache()
    print("seconds per review, including model load:", timings, flush=True)

    OUTPUT.write_text(json.dumps({"timings": timings, "rows": rows.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    HfApi().upload_file(path_or_fileobj=str(OUTPUT), path_in_repo=f"smoke/{OUTPUT.name}", repo_id=DATASET_REPO,
                        repo_type="dataset", commit_message="Translation model comparison")
    print(f"pushed: https://huggingface.co/datasets/{DATASET_REPO}/tree/main/smoke/")


if __name__ == "__main__":
    main()
