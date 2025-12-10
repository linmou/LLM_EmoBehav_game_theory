#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"

CONFIGS=(
  "config/fantom_nogen_qwen3_nothink.yaml"
  "config/fantom_nogen_qwen3_think.yaml"
)

for CONFIG_PATH in "${CONFIGS[@]}"; do
  echo "\n>>> Running $CONFIG_PATH with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m emotion_experiment_engine.emotion_experiment_series_runner --config "$CONFIG_PATH"
  echo ">>> Finished $CONFIG_PATH"

done
