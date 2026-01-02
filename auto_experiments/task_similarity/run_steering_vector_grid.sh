#!/usr/bin/env bash

# Loop over models, seeds, and intensities to generate steering vectors
# using auto_experiments.task_similarity.run_pd_defection_experiment.
#
# Usage:
#   bash auto_experiments/task_similarity/run_steering_vector_grid.sh
#
# Notes:
# - Uses CUDA_VISIBLE_DEVICES=1,2 and conda env llm_fresh via conda run.
# - Seeds are 0..29 (30 seeds total).

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2

OUTPUT_DIR="auto_experiments/task_similarity/results/steering_vectors"
MAX_LENGTH=512
BATCH_SIZE=16

# Qwen2.5 model paths
MODELS=(
  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct"
  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct"
)

INTENSITIES=(1.5)

for MODEL in "${MODELS[@]}"; do
  for SEED in $(seq 20 29); do
    for INTEN in "${INTENSITIES[@]}"; do
      echo "=== Running steering vector experiment ==="
      echo "Model:      ${MODEL}"
      echo "Seed:       ${SEED}"
      echo "Intensity:  ${INTEN}"
      echo "Output dir: ${OUTPUT_DIR}"
      echo "-----------------------------------------"

      conda run -n llm_fresh python -m auto_experiments.task_similarity.run_pd_defection_experiment \
        --model "${MODEL}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --intensity "${INTEN}" \
        --middle_third_only

      echo
    done
  done
done

