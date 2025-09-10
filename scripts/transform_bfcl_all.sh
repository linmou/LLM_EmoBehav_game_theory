#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to transform BFCL live_simple and live_multiple in one go.
# Usage:
#   scripts/transform_bfcl_all.sh /path/to/bfcl_eval/data
# Expects files:
#   <SRC>/BFCL_v4_live_simple.json
#   <SRC>/possible_answer/BFCL_v4_live_simple.json
#   <SRC>/BFCL_v4_live_multiple.json
#   <SRC>/possible_answer/BFCL_v4_live_multiple.json

SRC_DIR=${1:-}
if [[ -z "${SRC_DIR}" ]]; then
  echo "Usage: $0 /path/to/bfcl_eval/data" >&2
  exit 1
fi

PY=python

${PY} scripts/bfcl_transform.py \
  --category live_simple \
  --input "${SRC_DIR}/BFCL_v4_live_simple.json" \
  --ground_truth "${SRC_DIR}/possible_answer/BFCL_v4_live_simple.json" \
  --output data/BFCL/bfcl_live_simple.jsonl

${PY} scripts/bfcl_transform.py \
  --category live_multiple \
  --input "${SRC_DIR}/BFCL_v4_live_multiple.json" \
  --ground_truth "${SRC_DIR}/possible_answer/BFCL_v4_live_multiple.json" \
  --output data/BFCL/bfcl_live_multiple.jsonl

echo "All BFCL transformations completed. Output in data/BFCL." 

