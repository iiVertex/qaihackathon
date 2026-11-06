# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6",
#   "outscraper==6.0.5",
#   "pandas==3.0.6",
#   "pyarrow==26.0.0",
#   "huggingface_hub==2.4.0",
# ]
# ///
"""Phase 2: scrape Google Maps reviews with Outscraper.

  test       1 business x 20 reviews. Shows 5 rows and the available fields.
  preflight  1 review per candidate, to read each place's real review count.
  full       Up to 50 newest + 50 lowest-rated reviews per selected business.
             Needs --confirm-paid.
             Stays local unless --push is passed.

The key is read from OUTSCRAPER_API_KEY and is never printed.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from outscraper import OutscraperClient

BUSINESSES_FILE = Path("businesses.json")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DATASET_REPO = "Hermesione/reviewere-dataset"

# Hard caps from the SPENDING RULES in CLAUDE.md.
TEST_REVIEWS = 20
MAX_REVIEWS_PER_BUSINESS = 200
MAX_BUSINESSES = 10

# "Average number of reviews": enough to fill the cap, not a tourist spotlight.
MIN_PLACE_REVIEWS = 200
MAX_PLACE_REVIEWS = 5000

# Outscraper pricing page: first 500 reviews free, then $3 per 1,000.
FREE_REVIEWS = 500
USD_PER_1000 = 3.0

# Lowest-rated reviews find complaints, but skew the sample. Rows record which
# sample they came from, so percentages are only computed on "newest".
SAMPLES = ("newest", "lowest_rating")

REQUIRED_FIELDS = ("review_id", "review_text", "review_rating", "review_timestamp")
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LATIN = re.compile(r"[A-Za-z]")


def client() -> OutscraperClient:
    key = os.environ.get("OUTSCRAPER_API_KEY")
    if not key:
        sys.exit("ERROR: OUTSCRAPER_API_KEY is not set. No API call was made.")
    return OutscraperClient(api_key=key)


def load_candidates() -> list[dict]:
    return json.loads(BUSINESSES_FILE.read_text(encoding="utf-8"))["candidates"]


def fetch(api: OutscraperClient, query: str, reviews_limit: int, sort: str = "newest") -> dict:
    """Fetch one place. reviews_limit=0 means unlimited in the API, so refuse it."""
    if not 1 <= reviews_limit <= MAX_REVIEWS_PER_BUSINESS:
        sys.exit(f"ERROR: reviews_limit {reviews_limit} is outside 1-{MAX_REVIEWS_PER_BUSINESS}.")
    result = api.google_maps_reviews(
        query, reviews_limit=reviews_limit, limit=1, sort=sort, ignore_empty=True, language="en"
    )
    if not result or not isinstance(result[0], dict):
        raise RuntimeError(f"Unexpected response for {query!r}: {str(result)[:200]}")
    return result[0]


def language_of(text: str) -> str:
    arabic, latin = len(ARABIC.findall(text)), len(LATIN.findall(text))
    if arabic == 0 and latin == 0:
        return "other"
    return "ar" if arabic >= latin else "en"


def to_rows(place: dict, category: str, sample: str = "newest") -> list[dict]:
    """Keep what the pipeline needs. Reviewer names and profile links are dropped."""
    rows = []
    for review in place.get("reviews_data") or []:
        author = str(review.get("author_id") or "")
        text = review.get("review_text") or ""
        rows.append({
            "review_id": review.get("review_id"),
            "business": place.get("name"),
            "category": category,
            "sample": sample,
            "google_id": place.get("google_id"),
            "place_rating": place.get("rating"),
            "place_reviews": place.get("reviews"),
            "rating": review.get("review_rating"),
            "text": text,
            "language": language_of(text),
            "timestamp_utc": pd.to_datetime(review.get("review_timestamp"), unit="s", utc=True),
            "author_hash": hashlib.sha256(author.encode()).hexdigest()[:16] if author else None,
            "author_review_count": review.get("author_reviews_count"),
        })
    return rows


def clean(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="review_id")
    df = df.drop_duplicates(subset=["google_id", "author_hash", "timestamp_utc", "text"])
    print(f"dedupe: {before} -> {len(df)} rows")
    return df


def save_raw(place: dict, label: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.json"
    path.write_text(json.dumps(place, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def cmd_test(args: argparse.Namespace) -> None:
    candidate = load_candidates()[0]
    print(f"TEST: {candidate['query']} x {TEST_REVIEWS} reviews")
    place = fetch(client(), candidate["query"], TEST_REVIEWS)
    print(f"raw saved: {save_raw(place, 'test_' + candidate['name'])}")

    reviews = place.get("reviews_data") or []
    print("\nplace:", {k: place.get(k) for k in ("name", "address", "type", "rating", "reviews", "google_id")})
    print(f"reviews returned: {len(reviews)}")
    print("\nplace fields:", sorted(k for k in place if k != "reviews_data"))
    if not reviews:
        sys.exit("ERROR: no reviews returned.")
    print("\nreview fields:", sorted(reviews[0]))
    missing = [f for f in REQUIRED_FIELDS if f not in reviews[0]]
    print("missing required fields:", missing or "none")
    translated = sum("translated by google" in (r.get("review_text") or "").lower() for r in reviews)
    print("reviews with a Google translation marker:", translated)

    if missing:
        sys.exit("ERROR: fix to_rows() before going further.")
    df = clean(to_rows(place, candidate["category"]))
    print("\nlanguage split:", df["language"].value_counts().to_dict())
    print("\nfirst 5 rows:")
    for row in df.head(5).itertuples():
        print(f"  {row.timestamp_utc:%Y-%m-%d} | {row.rating} stars | {row.language} | {row.text[:120]!r}")


def cmd_preflight(args: argparse.Namespace) -> None:
    candidates = load_candidates()
    print(f"PREFLIGHT: {len(candidates)} candidates x 1 review each")
    api = client()
    results = []
    for candidate in candidates:
        place = fetch(api, candidate["query"], 1)
        count = place.get("reviews")
        results.append({
            "candidate": candidate["name"],
            "category": candidate["category"],
            "matched_name": place.get("name"),
            "address": place.get("address"),
            "rating": place.get("rating"),
            "reviews": count,
            "in_band": isinstance(count, int) and MIN_PLACE_REVIEWS <= count <= MAX_PLACE_REVIEWS,
        })
        print(results[-1])
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "preflight.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {path}. In band ({MIN_PLACE_REVIEWS}-{MAX_PLACE_REVIEWS}): "
          f"{sum(r['in_band'] for r in results)}/{len(results)}")


def cmd_full(args: argparse.Namespace) -> None:
    selected = [c for c in load_candidates() if c.get("selected")]
    if not selected:
        sys.exit("ERROR: no businesses have selected=true in businesses.json.")
    if len(selected) > MAX_BUSINESSES:
        sys.exit(f"ERROR: {len(selected)} selected, hard cap is {MAX_BUSINESSES}.")

    limit = min(args.reviews, MAX_REVIEWS_PER_BUSINESS // len(SAMPLES))
    planned = len(selected) * limit * len(SAMPLES)
    cost = max(planned - FREE_REVIEWS, 0) * USD_PER_1000 / 1000
    print(f"PLAN: {len(selected)} businesses x up to {limit} reviews x {len(SAMPLES)} samples = up to {planned} reviews, "
          f"about ${cost:.2f} if the free {FREE_REVIEWS} are unused")
    if not args.confirm_paid:
        sys.exit("Stopped: pass --confirm-paid after approval. No API call was made.")

    api = client()
    rows = []
    for candidate in selected:
        for sample in SAMPLES:
            place = fetch(api, candidate["query"], limit, sort=sample)
            save_raw(place, f"{candidate['name']}_{sample}")
            reviews = place.get("reviews_data") or []
            missing = [f for f in REQUIRED_FIELDS if reviews and f not in reviews[0]]
            if missing:
                sys.exit(f"ERROR: {candidate['name']} is missing fields {missing}. Stopping.")
            rows += to_rows(place, candidate["category"], sample)
            print(f"{candidate['name']} [{sample}]: {len(reviews)} reviews")

    df = clean(rows)
    parquet = DATA_DIR / "scraped_reviews.parquet"
    df.to_parquet(parquet, index=False)
    print(f"\nsaved {parquet}: {len(df)} rows")
    print("\nrows per business:\n", pd.crosstab(df["business"], df["sample"]).to_string())
    print("\nlanguage split:", df["language"].value_counts().to_dict())

    if args.push:
        from huggingface_hub import HfApi

        hub = HfApi()
        hub.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=str(RAW_DIR),
                          path_in_repo="raw", commit_message="Phase 2: raw Outscraper reviews")
        hub.upload_file(repo_id=DATASET_REPO, repo_type="dataset", path_or_fileobj=str(parquet),
                        path_in_repo=parquet.name, commit_message="Phase 2: clean reviews parquet")
        print(f"pushed: https://huggingface.co/datasets/{DATASET_REPO}")
    else:
        print("\nNot pushed (no --push).")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("test", help="1 business x 20 reviews").set_defaults(run=cmd_test)
    commands.add_parser("preflight", help="1 review per candidate").set_defaults(run=cmd_preflight)
    full = commands.add_parser("full", help="selected businesses, needs --confirm-paid")
    full.add_argument("--reviews", type=int, default=50, help="per sample, per business")
    full.add_argument("--confirm-paid", action="store_true")
    full.add_argument("--push", action="store_true", help="upload results to the dataset repo")
    full.set_defaults(run=cmd_full)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
