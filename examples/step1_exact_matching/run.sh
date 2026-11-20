#!/usr/bin/env bash
# The Step 1 worked example from docs/guide_step1_exact_matching.md, end to end.
# Run from this directory with biosemalign[lexicon] installed. Every code here is invented.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf bundle out
mkdir -p out

# 1. Compile the dictionary tables into a pinned bundle.
biosemalign lexicon build --source tables --declaration tables/vocabularies.yaml --output bundle
biosemalign lexicon info bundle

# 2. Point BioSemAlign at it. FROZEN (the default mode) requires the digest pin.
export BIOSEMALIGN_LEXICON_PATH="$PWD/bundle"
export BIOSEMALIGN_LEXICON_SHA256="$(python -c 'import json; print(json.load(open("bundle/manifest.json"))["bundle_sha256"])')"
export BIOSEMALIGN_CACHE_DIR="$PWD/out/cache"

# 3. Step 1 over the batch.
biosemalign align-batch -p step1_exact.yaml -i mentions.csv -o out/step1_crosswalk.csv \
  --value-column drugname --id-columns report_id --id-columns drug_seq \
  --context-columns route --memo out/step1_memo.sqlite

# 4. Check the artifacts and score them against the gold standard.
biosemalign verify -i out/step1_crosswalk.jsonl -p step1_exact.yaml
biosemalign evaluate --gold gold.jsonl --predictions out/step1_crosswalk.jsonl \
  --report out/step1_eval.json > /dev/null
python - <<'PY'
import json
report = json.load(open("out/step1_eval.json"))
print({k: report[k] for k in ("gold_cases", "auto_accepted", "reviewed", "coverage", "auto_accept_precision")})
PY
