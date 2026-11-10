#!/usr/bin/env bash
# Step 2: the minimum set (A1, A2, A3) on gpt56luna, one run after another.
# Billed. Run inside tmux:  bash epai/ai_detection/rebuttal/run_main.sh
# Resumable: re-running skips finished rows. Extra args (e.g. --workers 4) are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).log"
run() {
  echo "================ $* ================" | tee -a "$LOG"
  .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna --yes "$@" 2>&1 | tee -a "$LOG"
}

run --condition base_hp4  "${@}"
run --condition base_free "${@}"
run --condition base_v3   "${@}"
run --condition lj_v3     "${@}"
run --condition base_v3 --subset near --reps 3 "${@}"
echo "done -- log: $LOG"
