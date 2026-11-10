#!/usr/bin/env bash
# Step 3: optional extras on gpt56luna -- A5 label order/wording, then A4 repeated
# BaseJudge at temperature 1. Billed. Run after run_main.sh, inside tmux:
#   bash epai/ai_detection/rebuttal/run_extra.sh
# Resumable. Extra args (e.g. --workers 4) are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_$(date +%Y%m%d_%H%M%S).log"
run() {
  echo "================ $* ================" | tee -a "$LOG"
  .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna --yes "$@" 2>&1 | tee -a "$LOG"
}

run --condition lj_swap    "${@}"
run --condition lj_alt     "${@}"
run --condition lj_letters "${@}"
run --condition base_v3_t1 "${@}"
echo "done -- log: $LOG"
