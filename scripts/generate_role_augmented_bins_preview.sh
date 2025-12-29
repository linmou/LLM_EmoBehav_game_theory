#!/usr/bin/env bash
set -euo pipefail

# Generate role-specific augmented choice bins (persisted JSON).
# Responsible for:
# - Trust_Game_Trustee (bins=5) using trustee_behavior_choices
# - Ultimatum_Game_Responder (bins=5) using responder_behavior_choices
#
# Usage:
#   scripts/generate_role_augmented_bins_preview.sh [limit]
#
# Notes:
# - Default limit is 500 (matches "preview_first500" configs).
# - Set WORKERS/MODEL via env vars to speed up/override.

LIMIT="${1:-500}"
WORKERS="${WORKERS:-8}"
MODEL="${MODEL:-gemini-2.5-flash}"

OUT_DIR="data_creation/scenario_creation/langgraph_creation/augmented_bins_preview"
mkdir -p "$OUT_DIR"

python scripts/augment_game_choice_bins_gemini.py \
  --input data_creation/scenario_creation/langgraph_creation/Trust_Game_Trustor_all_data_samples.json \
  --output "$OUT_DIR/Trust_Game_Trustee_all_data_samples.aug_bins5.${MODEL}.first${LIMIT}.json" \
  --bins 5 \
  --model "$MODEL" \
  --choice-field trustee_behavior_choices \
  --limit "$LIMIT" \
  --workers "$WORKERS"

python scripts/augment_game_choice_bins_gemini.py \
  --input data_creation/scenario_creation/langgraph_creation/Ultimatum_Game_Proposer_all_data_samples.json \
  --output "$OUT_DIR/Ultimatum_Game_Responder_all_data_samples.aug_bins5.${MODEL}.first${LIMIT}.json" \
  --bins 5 \
  --model "$MODEL" \
  --choice-field responder_behavior_choices \
  --limit "$LIMIT" \
  --workers "$WORKERS"

echo "Wrote:"
echo "  $OUT_DIR/Trust_Game_Trustee_all_data_samples.aug_bins5.${MODEL}.first${LIMIT}.json"
echo "  $OUT_DIR/Ultimatum_Game_Responder_all_data_samples.aug_bins5.${MODEL}.first${LIMIT}.json"

