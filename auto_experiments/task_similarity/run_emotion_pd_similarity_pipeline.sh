#!/usr/bin/env bash
set -euo pipefail

# Runs:
# 1) similarity (emotion vs PD) on PD prompts (default: all samples)
# 2) decision-impact join with a results folder
# 3) summary (top |pearson r| + last layers)
#
# Usage:
#   bash auto_experiments/task_similarity/run_emotion_pd_similarity_pipeline.sh \
#     --result_dir results/new_game_theory/..._20251229_072025
#
# This script reads `model_path`, `emotions`, and `intensities` from
# `<result_dir>/experiment_config.json` and only logs them (no need to pass as args).
# For PD defection vectors, you can point at a specific PD vector run via:
#   --pd_vectors_dir .../layer_vectors
#   --split_manifest .../split_manifest.json

RESULT_DIR=""
MAX_LENGTH="1024"
BATCH_SIZE="60"
DEVICE_MAP="auto"
SPLIT="all"
PD_VECTORS_DIR=""
SPLIT_MANIFEST=""
MODEL_PATH_OVERRIDE=""
EMOTION_REP_READER=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result_dir) RESULT_DIR="$2"; shift 2;;
    --model_path) MODEL_PATH_OVERRIDE="$2"; shift 2;;
    --max_length) MAX_LENGTH="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --device_map) DEVICE_MAP="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --run_id) RUN_ID="$2"; shift 2;;
    --pd_vectors_dir) PD_VECTORS_DIR="$2"; shift 2;;
    --split_manifest) SPLIT_MANIFEST="$2"; shift 2;;
    --emotion_rep_reader) EMOTION_REP_READER="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$RESULT_DIR" ]]; then
  echo "--result_dir is required" >&2
  exit 2
fi
if [[ ! -f "$RESULT_DIR/experiment_config.json" ]]; then
  echo "Missing $RESULT_DIR/experiment_config.json" >&2
  exit 2
fi

# conda activate scripts are not `set -u` safe (they reference unset vars).
set +u
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate llm_fresh
set -u

IFS=$'\t' read -r MODEL_PATH EMOTIONS_CSV INTENSITIES <<<"$(
  python - "$RESULT_DIR" <<'PY'
import sys
from pathlib import Path

from auto_experiments.task_similarity.pipeline_config_reader import read_pipeline_config

result_dir = Path(sys.argv[1])
cfg = read_pipeline_config(result_dir / "experiment_config.json")
intensities = ",".join(str(x) for x in cfg.intensities)
emotions = ",".join(cfg.emotions)
print(f"{cfg.model_path}\t{emotions}\t{intensities}")
PY
)"

echo "experiment_config.json:" >&2
echo "  model_path: $MODEL_PATH" >&2
echo "  emotions:   $EMOTIONS_CSV" >&2
echo "  intensities:$INTENSITIES" >&2
echo "  split:      $SPLIT" >&2

MODEL_PATH_SELECTED="$MODEL_PATH"
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
  MODEL_PATH_SELECTED="$MODEL_PATH_OVERRIDE"
fi

if [[ ! -d "$MODEL_PATH_SELECTED" ]]; then
  # Common case: experiment_config.json was produced on another filesystem.
  if [[ -z "$MODEL_PATH_OVERRIDE" && "$MODEL_PATH" == *"/huggingface_models/"* ]]; then
    SUFFIX="${MODEL_PATH#*"/huggingface_models/"}"
    CAND="/data/home/jjl7137/huggingface_models/$SUFFIX"
    if [[ -d "$CAND" ]]; then
      echo "  model_path_resolved: $CAND (mapped from experiment_config.json)" >&2
      MODEL_PATH_SELECTED="$CAND"
    fi
  fi
fi

if [[ ! -d "$MODEL_PATH_SELECTED" ]]; then
  echo "Model path not found: $MODEL_PATH_SELECTED" >&2
  echo "Pass a local path via: --model_path /data/home/.../huggingface_models/..." >&2
  exit 2
fi

if [[ -z "$EMOTION_REP_READER" ]]; then
  # Try to auto-select a RepReader pickle matching this model's hidden size + layer count.
  # Convention: look in the main repo's representation_storage folder if it exists.
  CAND_BASE="/data/home/jjl7137/LLM_EmoBehav_game_theory/neuro_manipulation/representation_storage"
  if [[ -d "$CAND_BASE" ]]; then
    EMOTION_REP_READER="$(
      python - "$MODEL_PATH_SELECTED" "$CAND_BASE" <<'PY'
import sys
from pathlib import Path

import numpy as np
from transformers import AutoConfig

model_path = sys.argv[1]
base = Path(sys.argv[2])

cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
hidden_size = int(getattr(cfg, "hidden_size"))
num_layers = int(getattr(cfg, "num_hidden_layers"))

best = None  # (mtime, path)
for p in sorted(base.glob("emotion_rep_reader_*.pkl")):
    try:
        import pickle
        raw = pickle.load(open(p, "rb"))
        if not isinstance(raw, dict) or "anger" not in raw:
            continue
        rr = raw["anger"]
        if not hasattr(rr, "directions"):
            continue
        dirs = rr.directions
        if len(dirs) != num_layers:
            continue
        k = next(iter(dirs))
        v = np.asarray(dirs[k])
        if v.shape[-1] != hidden_size:
            continue
        mtime = p.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, p)
    except Exception:
        continue

if best is None:
    sys.exit(2)
print(str(best[1]))
PY
    )" || true
  fi
fi

if [[ -z "$EMOTION_REP_READER" || ! -f "$EMOTION_REP_READER" ]]; then
  echo "No compatible emotion RepReader found for this model." >&2
  echo "Provide one via: --emotion_rep_reader /path/to/emotion_rep_reader_XXXX.pkl" >&2
  exit 2
fi
echo "  emotion_rep_reader: $EMOTION_REP_READER" >&2

if [[ -z "$PD_VECTORS_DIR" || -z "$SPLIT_MANIFEST" ]]; then
  MODEL_NAME="$(basename "$MODEL_PATH_SELECTED")"
  BASE="auto_experiments/task_similarity/results/steering_vectors/$MODEL_NAME"
  if [[ -d "$BASE" ]]; then
    CHOSEN_SEED_DIR="$(find "$BASE" -type d -path '*/seed_20' 2>/dev/null | sort | tail -n 1)"
    if [[ -z "$CHOSEN_SEED_DIR" ]]; then
      CHOSEN_SEED_DIR="$(find "$BASE" -type f -name split_manifest.json -printf '%h\n' 2>/dev/null | sort | tail -n 1)"
    fi
    if [[ -n "$CHOSEN_SEED_DIR" ]]; then
      PD_VECTORS_DIR="${PD_VECTORS_DIR:-$CHOSEN_SEED_DIR/layer_vectors}"
      SPLIT_MANIFEST="${SPLIT_MANIFEST:-$CHOSEN_SEED_DIR/split_manifest.json}"
      echo "  pd_vectors_dir: $PD_VECTORS_DIR" >&2
      echo "  split_manifest: $SPLIT_MANIFEST" >&2
    fi
  fi
fi

EXTRA_PD_ARGS=()
EXTRA_PD_ARGS+=(--emotion_rep_reader "$EMOTION_REP_READER")
if [[ -n "$PD_VECTORS_DIR" ]]; then
  EXTRA_PD_ARGS+=(--pd_vectors_dir "$PD_VECTORS_DIR")
fi
if [[ -n "$SPLIT_MANIFEST" ]]; then
  EXTRA_PD_ARGS+=(--split_manifest "$SPLIT_MANIFEST")
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

OUTPUT_ROOT="auto_experiments/task_similarity/results/anger_pd_delta_similarity"
RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_ROOT"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
cat > "$RUN_ROOT/config.json" <<EOF
{
  "run_id": "$RUN_ID",
  "git_commit": "${GIT_COMMIT}",
  "result_dir": "${RESULT_DIR}",
  "model_path_selected": "${MODEL_PATH_SELECTED}",
  "emotions": "${EMOTIONS_CSV}",
  "intensities": "${INTENSITIES}",
  "split": "${SPLIT}",
  "max_length": ${MAX_LENGTH},
  "batch_size": ${BATCH_SIZE},
  "device_map": "${DEVICE_MAP}",
  "emotion_rep_reader": "${EMOTION_REP_READER}",
  "pd_vectors_dir": "${PD_VECTORS_DIR}",
  "split_manifest": "${SPLIT_MANIFEST}"
}
EOF
echo "run_root: $RUN_ROOT" >&2

IFS=',' read -ra EMO_ARR <<<"$EMOTIONS_CSV"
for EMO in "${EMO_ARR[@]}"; do
  echo "=== emotion: $EMO ===" >&2

  SIM_DIR="$(
    python -m auto_experiments.task_similarity.emotion_pd_delta_similarity \
      --emotion "$EMO" \
      --split "$SPLIT" \
      --run_id "$RUN_ID" \
      --intensities "$INTENSITIES" \
      --model "$MODEL_PATH_SELECTED" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE" \
      --device_map "$DEVICE_MAP" \
      "${EXTRA_PD_ARGS[@]}"
  )"

  python -m auto_experiments.task_similarity.analyze_similarity_decision_impact \
    --similarity_run_dir "$SIM_DIR" \
    --result_dir "$RESULT_DIR" \
    --emotion "$EMO"

  python -m auto_experiments.task_similarity.summarize_similarity_decision_impact \
    --impact_dir "$SIM_DIR/decision_impact/$EMO" \
    --top_k 10 \
    --last_k 5
done
