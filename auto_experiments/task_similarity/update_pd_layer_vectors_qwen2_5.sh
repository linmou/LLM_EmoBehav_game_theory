#!/usr/bin/env bash
set -euo pipefail

# Update PD layer vectors for all Qwen2.5 models.
# This script should work when invoked from any directory.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT}/auto_experiments/task_similarity/results"

#source /usr/local/anaconda3/etc/profile.d/conda.sh
#conda activate llm_fresh
#
MODELS=(
  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct"
  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct"
#  "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  echo "=== Running PD defection experiment for: ${MODEL} ==="
  python -m auto_experiments.task_similarity.run_pd_defection_experiment \
    --model "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --max_length 256 \
    --batch_size 8 \
    --seed 0 \
    --intensity 1.0 \
    --middle_third_only
done

