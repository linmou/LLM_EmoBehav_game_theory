#!/usr/bin/env bash
set -euo pipefail

python -m auto_experiments.sc2_dataset.generate_sc2_scenarios_from_intent_map_with_gemini \
  --game_name Sealed_Auction \
  --intent_map_jsonl /data/home/jjl7137/MSC/datasets/intent_map_dataset_air72_base96p6_drop94p6_gold50.jsonl \
  --out data_creation/scenario_creation/langgraph_creation/SC2_Sealed_Auction_all_data_samples.json \
  --concurrency 20 \
  "$@"
