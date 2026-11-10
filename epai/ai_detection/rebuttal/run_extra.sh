#!/usr/bin/env bash
# Step 3: optional extras on gpt56luna -- A5 label order/wording, then A4 repeated
# BaseJudge at temperature 1. Billed (~$11). Run after run_main.sh, inside tmux:
#   bash epai/ai_detection/rebuttal/run_extra.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=4; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition lj_swap
run --condition lj_alt
run --condition lj_letters
run --condition base_v3_t1
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
