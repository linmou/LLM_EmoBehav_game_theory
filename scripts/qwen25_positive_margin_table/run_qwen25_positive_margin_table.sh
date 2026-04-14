#!/usr/bin/env bash
# Purpose: reproduce the Qwen2.5 positive-margin table in one command by delegating to the lightweight Python orchestrator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source /home/jjl7137/anaconda3/etc/profile.d/conda.sh
conda activate llm

set -euo pipefail

python "${PROJECT_ROOT}/scripts/qwen25_positive_margin_table/run_qwen25_positive_margin_table.py" "$@"
