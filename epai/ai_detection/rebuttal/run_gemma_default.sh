#!/usr/bin/env bash
# R4: A1 precision on gemma4-31b with the paper's BaseJudge call (structured answer,
# endpoint defaults), four decimals and no limit, on humans_plus_third (no cost,
# ~4 h each with the endpoint's 4 concurrent calls). Inside tmux:
#   bash epai/ai_detection/rebuttal/run_gemma_default.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gemma_default_$(date +%Y%m%d_%H%M%S).log"
MODEL=gemma4-31b; TOTAL=2; EXTRA_ARGS=(--workers 4 "$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_default_hp4 --subset humans_plus_third
run --condition base_default_free --subset humans_plus_third
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
