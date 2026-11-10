#!/usr/bin/env bash
# Step 4: gemma4-31b replication of A1/A2 on the institutional endpoint (no cost).
# The endpoint serves 4 calls at a time, so these run one after another with 4 workers.
# Run where the endpoint is reachable, inside tmux:
#   bash epai/ai_detection/rebuttal/run_gemma.sh
# Resumable. Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gemma_$(date +%Y%m%d_%H%M%S).log"
run() {
  echo "================ $* ================" | tee -a "$LOG"
  .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gemma4-31b --workers 4 --yes "$@" 2>&1 | tee -a "$LOG"
}

run --condition base_hp4  "${@}"
run --condition base_free "${@}"
run --condition base_v3   "${@}"
echo "done -- log: $LOG"
