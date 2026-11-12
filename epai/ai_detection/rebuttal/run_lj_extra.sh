#!/usr/bin/env bash
# LogJudge counterparts of the paper-call BaseJudge reruns, on gpt56luna:
#   L1  A4 averaging: lj_v3 (the paper's LogJudge call) on `small` x 3 (6,200 texts),
#       the same rows as base_default's repeats -> k calls vs k calls
#   L2  A1: lj_default (LogJudge prompt + logprobs, endpoint default settings, so luna
#       may reason) on all 36,148 texts
# Billed (L1 ~$5; L2 ~$10-15 depending on reasoning). Probe L2 first: it must print a
# label token with a probability window. If it FAILs or shows no logprobs, the endpoint
# does not give logprobs with reasoning: comment out the second `run` line below.
#   .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
#       --condition lj_default --probe 12
# Then, inside tmux:
#   bash epai/ai_detection/rebuttal/run_lj_extra.sh
# Resumable; failed rows are retried automatically (_common.sh). Extra args are passed on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/lj_extra_$(date +%Y%m%d_%H%M%S).log"
MODEL=gpt56luna; TOTAL=2; EXTRA_ARGS=("$@")
source epai/ai_detection/rebuttal/_common.sh

run --condition lj_v3 --subset small --reps 3
run --condition lj_default --subset full
echo "================ all $TOTAL steps done $(date '+%H:%M:%S') -- log: $LOG ================"
