#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the Python entrypoint:
# `python -m auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline`.

# conda activate scripts are not `set -u` safe (they reference unset vars).
set +u
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate llm_fresh
set -u

python -m auto_experiments.task_similarity.run_emotion_pd_similarity_pipeline "$@"
