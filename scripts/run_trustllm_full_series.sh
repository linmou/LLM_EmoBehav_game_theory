#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"

CONFIGS=(
  "config/trustllm_ethics_full.yaml"
  "config/trustllm_fairness_full.yaml"
  "config/trustllm_privacy_full.yaml"
  "config/trustllm_robustness_full.yaml"
  "config/trustllm_safety_full.yaml"
  "config/trustllm_truthfulness_full.yaml"
)

for CONFIG_PATH in "${CONFIGS[@]}"; do
  echo "\n>>> Running $CONFIG_PATH with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m emotion_experiment_engine.emotion_experiment_series_runner --config "$CONFIG_PATH"
  echo ">>> Finished $CONFIG_PATH"

done
