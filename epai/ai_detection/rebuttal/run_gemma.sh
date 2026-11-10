#!/usr/bin/env bash
# Step 4: gemma4-31b replication of A1/A2 on the institutional endpoint (no cost).
# The endpoint serves 4 calls at a time, so these run one after another with 4 workers
# (~4 h each on humans_plus_third). Run where the endpoint is reachable, inside tmux:
#   bash epai/ai_detection/rebuttal/run_gemma.sh
# Resumable; failed rows are retried automatically, with pauses of 1, 5 and 15 min
# (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gemma_$(date +%Y%m%d_%H%M%S).log"
MODEL=gemma4-31b; TOTAL=3; EXTRA_ARGS=(--workers 4 "$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_hp4
run --condition base_free
run --condition base_v3
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
