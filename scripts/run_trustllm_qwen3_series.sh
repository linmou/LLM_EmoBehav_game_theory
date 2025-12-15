#!/usr/bin/env bash

set -euo pipefail

# Default to the sample device list if CUDA_VISIBLE_DEVICES is undefined
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"

CONFIGS=(
  "config/trustllm_ethics_qwen3_nothink.yaml"
  "config/trustllm_ethics_qwen3_think.yaml"
  "config/trustllm_fairness_qwen3_nothink.yaml"
  "config/trustllm_fairness_qwen3_think.yaml"
  "config/trustllm_privacy_qwen3_nothink.yaml"
  "config/trustllm_privacy_qwen3_think.yaml"
  "config/trustllm_robustness_qwen3_nothink.yaml"
  "config/trustllm_robustness_qwen3_think.yaml"
  "config/trustllm_safety_qwen3_nothink.yaml"
  "config/trustllm_safety_qwen3_think.yaml"
  "config/trustllm_truthfulness_qwen3_nothink.yaml"
  "config/trustllm_truthfulness_qwen3_think.yaml"
)

for CONFIG_PATH in "${CONFIGS[@]}"; do
  echo "\n>>> Running $CONFIG_PATH with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m emotion_experiment_engine.emotion_experiment_series_runner --config "$CONFIG_PATH"
  echo ">>> Finished $CONFIG_PATH"

done
