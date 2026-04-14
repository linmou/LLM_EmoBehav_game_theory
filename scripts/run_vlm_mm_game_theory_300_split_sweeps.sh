#!/usr/bin/env bash
# Purpose: launch two parallel 2-GPU sweeps for the multimodal game-theory 300-sample experiment and log GPU usage for each sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/run_logs"
mkdir -p "$LOG_DIR"

source /home/jjl7137/anaconda3/etc/profile.d/conda.sh
conda activate llm-test

STAMP="$(date +%Y%m%d_%H%M%S)"
HOUR_STAMP="$(date +%Y%m%d_%H)"
GPU01_MON_LOG="$LOG_DIR/gpu_monitor_vlm_mm_game_theory_300_gpu01_${STAMP}.log"
GPU23_MON_LOG="$LOG_DIR/gpu_monitor_vlm_mm_game_theory_300_gpu23_${STAMP}.log"
GPU01_RUN_LOG="$LOG_DIR/vlm_mm_game_theory_300_gpu01_${STAMP}.log"
GPU23_RUN_LOG="$LOG_DIR/vlm_mm_game_theory_300_gpu23_${STAMP}.log"
GPU01_REPORT="$ROOT_DIR/results/vlm_mm_game_theory/sample300_gpu01/memory_experiment_series_gpu01_${HOUR_STAMP}_memory_experiment_report.json"
GPU23_REPORT="$ROOT_DIR/results/vlm_mm_game_theory/sample300_gpu23/memory_experiment_series_gpu23_${HOUR_STAMP}_memory_experiment_report.json"
MERGED_OUTPUT_DIR="$ROOT_DIR/results/vlm_mm_game_theory/sample300_merged"

(
  while true; do
    echo "=== $(date --iso-8601=seconds) ==="
    nvidia-smi -i 0,1 --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
    echo
    sleep 30
  done
) >"$GPU01_MON_LOG" 2>&1 &
GPU01_MON_PID=$!

(
  while true; do
    echo "=== $(date --iso-8601=seconds) ==="
    nvidia-smi -i 2,3 --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
    echo
    sleep 30
  done
) >"$GPU23_MON_LOG" 2>&1 &
GPU23_MON_PID=$!

cleanup() {
  kill "$GPU01_MON_PID" "$GPU23_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0,1 python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config "$ROOT_DIR/config/vlm_mm_game_theory_300_gpu01.yaml" \
  --name "memory_experiment_series_gpu01" >"$GPU01_RUN_LOG" 2>&1 &
GPU01_RUN_PID=$!

CUDA_VISIBLE_DEVICES=2,3 python -m emotion_experiment_engine.emotion_experiment_series_runner \
  --config "$ROOT_DIR/config/vlm_mm_game_theory_300_gpu23.yaml" \
  --name "memory_experiment_series_gpu23" >"$GPU23_RUN_LOG" 2>&1 &
GPU23_RUN_PID=$!

python -m emotion_experiment_engine.resource_recursive_workflow launch-eval-watchers \
  --report "$GPU01_REPORT" \
  --report "$GPU23_REPORT" \
  --env-name llm-test \
  --poll-interval-secs 30 \
  --max-workers 8 \
  --session-name-prefix vlm_mm_game_theory_eval >"$LOG_DIR/vlm_mm_game_theory_eval_sessions_${STAMP}.log"

echo "gpu01 monitor log: $GPU01_MON_LOG"
echo "gpu23 monitor log: $GPU23_MON_LOG"
echo "gpu01 run log: $GPU01_RUN_LOG"
echo "gpu23 run log: $GPU23_RUN_LOG"
echo "gpu01 report: $GPU01_REPORT"
echo "gpu23 report: $GPU23_REPORT"
echo "gpu01 run pid: $GPU01_RUN_PID"
echo "gpu23 run pid: $GPU23_RUN_PID"

wait "$GPU01_RUN_PID" "$GPU23_RUN_PID"

python -m emotion_experiment_engine.resource_recursive_workflow wait-and-merge \
  --report "$GPU01_REPORT" \
  --report "$GPU23_REPORT" \
  --merged-output-dir "$MERGED_OUTPUT_DIR" \
  --merged-series-name "memory_experiment_series_split_merged" \
  --poll-interval-secs 30 \
  --max-workers 8
