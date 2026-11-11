#!/usr/bin/env bash
# A3b: the paper's BaseJudge call (default settings, structured answer), repeated 3x
# on the `near` rows, so the paper's FusionJudge can be recomputed per capture
# together with lj_v3's repeats. Billed: probe first, it prints the cost per call:
#   .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
#       --condition base_default --probe 12
# Then, inside tmux:
#   bash epai/ai_detection/rebuttal/run_default.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/default_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=1; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition base_default
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
