#!/usr/bin/env bash
# Step 1: probes -- 12 gpt56luna calls per condition, printed, nothing saved.
# Run from anywhere:  bash epai/ai_detection/rebuttal/run_probes.sh
# Output is also written to data/parquet/tmp/rebuttal/logs/probes_<time>.log
set -uo pipefail   # no -e: one failing probe should not stop the rest
cd "$(dirname "$0")/../../.."
LOG_DIR=data/parquet/tmp/rebuttal/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/probes_$(date +%Y%m%d_%H%M%S).log"

for cond in base_free base_hp4 base_v3 lj_alt lj_letters; do
  echo "================ PROBE $cond ================" | tee -a "$LOG"
  .venv/bin/python -m epai.ai_detection.rebuttal.capture \
      --model gpt56luna --condition "$cond" --probe 12 2>&1 | tee -a "$LOG"
done
echo "done -- log: $LOG"
