#!/usr/bin/env bash
# LogJudge counterpart of the paper-call BaseJudge averaging (A4), on gpt56luna:
#   L1  lj_v3 (the paper's LogJudge call) on `small` x 3 (6,200 texts), the same rows
#       as base_default's repeats -> k calls vs k calls. Billed, ~$5.
# (A LogJudge with reasoning is impossible on luna: the endpoint refuses logprobs on
# reasoning calls -- see RUNBOOK.md.) Inside tmux:
#   bash epai/ai_detection/rebuttal/run_lj_extra.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/lj_extra_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=1; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition lj_v3 --subset small --reps 3
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
