#!/usr/bin/env bash
# A7: gemma4-31b BaseJudge (the paper's call) with its optional thinking mode ON, at two
# and four decimals, on `small` (6,200 texts). Free; thinking calls take ~13 s, so
# ~5-6 h per step with the endpoint's 4 concurrent calls. Probe first (2 calls each):
#   .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gemma4-31b \
#       --condition base_think --probe 4
# Then, inside tmux:
#   bash epai/ai_detection/rebuttal/run_gemma_think.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gemma_think_$(date +%Y%m%d_%H%M%S).log"
MODEL=gemma4-31b; TOTAL=2; EXTRA_ARGS=(--workers 4 "$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_think --subset small
run --condition base_think_hp4 --subset small
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
