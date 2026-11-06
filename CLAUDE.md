# Review Inspector — QAI x Hugging Face hackathon

Build tonight: a Gradio Hugging Face Space that scores reviews of 8–10 Qatar businesses for credibility, extracts likely violations with evidence, and gives an inspector a ranked brief + checklist. Heavy work is precomputed; the Space reads cached results and has one live "paste a review" box.

## Architecture (6 steps)
1. Ingest: Outscraper Google Maps reviews → dedupe, normalize timestamps, keep Arabic + English.
2. Credibility (0–1, never a "fake" verdict): behaviour signals (author has 1 review, 5-star bursts in 48h, near-duplicate text, rating vs text mismatch) + AI-text classifier (xlm-roberta-base fine-tuned on `MichiganNLP/MAiDE-up`).
3. Violations: `fastino/GLiNER2.5-multi-Decide` via `gliner2[local]`, zero-shot multi-label + evidence spans. CPU only.
4. Severity + brief: severity = category weight × confidence × credible review count × recency. Safety/hygiene hits are NEVER dropped, only marked low-confidence. Brief = LLM via HF Inference Providers, template fallback.
5. Cache: private HF Dataset repo with `reviews.parquet`, `businesses.parquet`, `mock_inspections.json`, `checklist_log/` (written via `huggingface_hub.CommitScheduler`).
6. Gradio Space: picker → risk card → complaints (safety first, evidence highlighted) → brief → checklist (Confirmed / False alarm / Resolved) → PDF export. Live box reruns steps 2–4. Models load lazily, only for the live box.

## Resources (fill in before starting)
- Namespace: `<ORG_OR_USER>`
- Demo Space: `<ORG_OR_USER>/review-inspector` (Gradio SDK, CPU upgrade hardware, $0.03/hr — set by a human, not ZeroGPU)
- Model repo: `<ORG_OR_USER>/maideup-xlmr` (private)
- Dataset repo: `<ORG_OR_USER>/review-inspector-data` (private)
- Secrets: `HF_TOKEN`, `OUTSCRAPER_API_KEY` come from environment variables / Space secrets. Never print, log, hardcode or commit them.

## Violation categories (GLiNER2 labels, tune wording in Phase 3)
- hygiene: food poisoning, pests, dirty kitchen or rooms, spoiled food, bad smells
- safety: fire hazards, blocked exits, injuries, broken structures, electrical danger
- pricing_fraud: overcharging, hidden fees, bill different from menu, refused refund
- licensing: unlicensed activity, alcohol or smoking violations, expired permits
- service_only: rude staff, slow service, no regulatory issue

## SPENDING RULES (hard — never break)
1. Paid compute ONLY through `hf jobs` with an explicit `--flavor` and `--timeout`. Allowed flavors: `cpu-basic`, `cpu-upgrade`, `t4-small`, `l4x1`. Max timeout 2h. Never A10G-large, A100, H100, or multi-GPU.
2. Before ANY paid action (a Job, an Inference Providers batch over 20 calls, an Outscraper call over 50 reviews) tell me: what it does, flavor, timeout, estimated cost in USD. Then WAIT for my explicit "yes".
3. Never change Space hardware, never create Spaces, Inference Endpoints, or persistent storage, never delete repos.
4. Every Job gets a smoke test first: same script on `cpu-basic`, `--timeout 10m`, ~200 rows. GPU only after the smoke test passes.
5. After launching a Job: give the job ID, follow its logs, and cancel it immediately (`hf jobs cancel <id>`) on any error. Confirm with `hf jobs ps` that nothing is left running.
6. Jobs are ephemeral: every script must push its outputs to the Hub before exiting, and log the repo URL.
7. Outscraper: first test 1 business × 20 reviews. Hard caps: 200 reviews per business, 10 businesses.
8. Space hardware stays on CPU upgrade. Do not add ZeroGPU code (`import spaces`, `@spaces.GPU`). If the live box is too slow, tell me; I change hardware myself.

## HONESTY RULES
- Never report a result you did not see in command output. Show the real output: row counts, metrics, sample predictions.
- No invented numbers, no synthetic or placeholder data unless I ask. Mock inspection records live only in `mock_inspections.json` and are labelled "sample data" in the UI.
- If something fails, say so and show the error. Do not silently swap in a fallback, a smaller model, or fake data.
- Do not guess APIs or CLI flags: check `--help`, docs, or package source first. Verify every model/dataset ID exists on the Hub before using it. Pin versions in `requirements.txt`.
- Metrics are only valid on a held-out split never used in training. Say which split.

## PHASES — do one at a time, stop and report after each, wait for "go"
0. Setup check (no writes): `hf auth whoami`, confirm the three repos exist with direct checks (`hf spaces info`, `hf models info`, `hf datasets info`, logged in, since private repos can look "not found" otherwise), show Space status and hardware, confirm `hf jobs --help` works. Do not use `hf repos list --namespace`: it calls an organizations endpoint and returns a false 404 on a personal account like Hermesione.
1. Space skeleton: README.md frontmatter (`sdk: gradio`, `app_file: app.py`), hello-world `app.py`, `requirements.txt`. Push. Confirm the Space reaches "Running".
2. Scrape: 1-business test → show 5 rows + available fields → wait → full scrape → raw JSON + clean parquet pushed to the dataset repo. Report rows per business and Arabic/English split.
3. GLiNER2 check: run on 30 Arabic + 30 English real reviews. Show a table (text, labels, confidence, spans). I judge Arabic quality; if poor, we translate Arabic to English before classification.
4. AI-text classifier: `train_detector.py` (uv inline dependencies), xlm-roberta-base on MAiDE-up, 1–2 epochs, max_len 256, fp16, eval on held-out test split, push model + metrics JSON. Smoke test on cpu-basic, then `l4x1` with `--timeout 90m` after my yes. Report accuracy/F1 per language, then % of 200 Qatar reviews flagged as AI-written.
5. Batch pipeline: `pipeline.py` (signals, credibility, GLiNER2, severity, brief with template fallback). Run as a Job on cpu-basic or cpu-upgrade. Push `reviews.parquet` + `businesses.parquet`. Show summary per business.
6. App: Gradio Blocks per step 6 above. Load parquet from the dataset repo, lazy-load models, checklist via CommitScheduler, PDF export. Push, then confirm every screen works on the live Space.

## Conventions
Python 3.11. Small focused files: `app.py`, `pipeline.py`, `scrape.py`, `train_detector.py`. Reviewer names and profile links never shown in the UI. Commit after each phase with a clear message.

## Resources
- Space: Hermesione/trial-review
- Model Repo: Hermesione/robert-fake-reviewer
- Dataset Repo: Hermesione/reviewere-dataset