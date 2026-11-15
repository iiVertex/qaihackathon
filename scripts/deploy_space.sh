#!/usr/bin/env bash
# Upload the app to the Hugging Face Space. Run from the repo root.
# The Space gets space/README.md as its README.md, because Spaces read their settings from it.
set -euo pipefail

SPACE="Hermesione/trial-review"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

cp app.py accounts.py i18n.py records.py report.py requirements.txt packages.txt "$stage"/
cp -r fonts "$stage"/
cp space/README.md "$stage"/README.md

hf upload "$SPACE" "$stage" . --repo-type space --commit-message "${1:-Update app}"
