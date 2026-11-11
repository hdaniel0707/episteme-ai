#!/usr/bin/env bash
# Reruns with the paper's BaseJudge call (default settings) on gpt56luna:
#   R1/R2  A1 precision: four decimals and no limit, all 36,148 texts
#   R3     A4 averaging: base_default on `small` x 3 (6,200 texts)
# Billed (~$30 at the probe's $0.00033/call). Probe first:
#   .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
#       --condition base_default_hp4 --probe 12        (and base_default_free)
# Then, inside tmux:
#   bash epai/ai_detection/rebuttal/run_default_extra.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/default_extra_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=3; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_default --subset small --reps 3
run --condition base_default_hp4 --subset full
run --condition base_default_free --subset full
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
