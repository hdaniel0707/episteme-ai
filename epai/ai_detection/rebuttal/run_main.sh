#!/usr/bin/env bash
# Step 2: the minimum set (A1, A2, A3) on gpt56luna, one run after another.
# Billed (~$36 at $0.20/M input tokens). Run inside tmux:
#   bash epai/ai_detection/rebuttal/run_main.sh
# Resumable: re-running skips finished rows. Failed rows are retried automatically
# (_common.sh). Extra args (e.g. --workers 4) are passed on.
# Progress from another terminal:  bash epai/ai_detection/rebuttal/status.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=5; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_hp4  --subset full
run --condition base_free --subset full
run --condition base_v3   --subset full
run --condition lj_v3
run --condition base_v3 --subset near --reps 3
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
