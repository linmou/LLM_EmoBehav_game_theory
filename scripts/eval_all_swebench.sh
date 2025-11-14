#!/usr/bin/env bash
set -euo pipefail

# Evaluate all SWE-bench runs under results/swebench using the dedicated
# acceptance helper. Keeps artifacts inside each run directory.
#
# Configurable env vars (override by exporting before running):
# - USE_CONDA: set to 1 to attempt conda activation (default: 0)
# - CONDA_SH: path to conda.sh for activation
# - CONDA_ENV: conda env to activate for the CLI (defaults to llm_fresh)
# - HARNESS_PY: Python 3.10+ interpreter to run the SWE-bench harness
#               (defaults to /home/jjl7137/.conda/envs/swebench310/bin/python)
# - SWE_REPO: path to SWE-bench repo (defaults to /data/home/jjl7137/SWE-bench)
# - DATASET: dataset name (defaults to SWE-bench/SWE-bench_Lite)
# - MAX_WORKERS: harness parallelism (defaults to 8)
# - DRY_RUN: set to 1 to print actions without running
#
# Usage:
#   bash scripts/eval_all_swebench.sh
#   DRY_RUN=1 bash scripts/eval_all_swebench.sh
#   MAX_WORKERS=4 HARNESS_PY=/home/jjl7137/.conda/envs/swebench311/bin/python \
#     bash scripts/eval_all_swebench.sh

USE_CONDA="${USE_CONDA:-0}"
CONDA_SH="${CONDA_SH:-/usr/local/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-llm_fresh}"
HARNESS_PY="${HARNESS_PY:-/home/jjl7137/.conda/envs/swebench310/bin/python}"
SWE_REPO="${SWE_REPO:-/data/home/jjl7137/SWE-bench}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Lite}"
MAX_WORKERS="${MAX_WORKERS:-32}"
DRY_RUN="${DRY_RUN:-0}"

echo "[INFO] HARNESS_PY=$HARNESS_PY"
echo "[INFO] SWE_REPO=$SWE_REPO"
echo "[INFO] DATASET=$DATASET  MAX_WORKERS=$MAX_WORKERS"

if [[ "$USE_CONDA" == "1" ]]; then
  if [[ -f "$CONDA_SH" ]]; then
    echo "[INFO] Activating conda env: $CONDA_ENV"
    # shellcheck disable=SC1090
    source "$CONDA_SH" || true
    conda activate "$CONDA_ENV" >/dev/null 2>&1 || true
  else
    echo "[WARN] CONDA_SH not found: $CONDA_SH (skipping activation)"
  fi
else
  echo "[INFO] Skipping conda activation (USE_CONDA=0)"
fi

root_dir="results/swebench"
if [[ ! -d "$root_dir" ]]; then
  echo "No directory: $root_dir" >&2
  exit 1
fi

shopt -s nullglob
mapfile -t runs < <(find "$root_dir" -maxdepth 1 -mindepth 1 -type d | sort)
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No runs found under $root_dir" >&2
  exit 0
fi

echo "[INFO] Found ${#runs[@]} runs under $root_dir"

for run_dir in "${runs[@]}"; do
  # Skip if a summary already exists
  if compgen -G "$run_dir/swebench_eval_summary.*.json" > /dev/null || [[ -f "$run_dir/summary_results.csv" ]]; then
    echo "[SKIP] $run_dir (summary exists)"
    continue
  fi

  cmd=(python -m emotion_experiment_engine.swebench_evaluation \
    --run-dir "$run_dir" \
    --swebench-repo "$SWE_REPO" \
    --dataset-name "$DATASET" \
    --python-executable "$HARNESS_PY" \
    --max-workers "$MAX_WORKERS")

  echo "[RUN ] ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  log="$run_dir/swebench_accept.log"
  set +e
  "${cmd[@]}" >>"$log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "[FAIL] $run_dir (exit $status) — see $log"
  else
    echo "[DONE] $run_dir — log: $log"
  fi
done
