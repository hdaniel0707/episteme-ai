# Sourced by run_main.sh / run_extra.sh / run_gemma.sh. Expects MODEL, TOTAL, LOG and
# optionally EXTRA_ARGS (array) to be set by the caller.
#
# run <capture args...>: one experiment, with automatic retry passes. capture.py exits 3
# when rows failed even after its own 3 per-call retries (seconds apart). Each retry pass
# re-sends only the failed rows, after a growing pause, which covers an endpoint that is
# down for minutes. Any other non-zero exit (a crash) stops the whole script.
STEP=0
PAUSES=(60 300 900)
run() {
  STEP=$((STEP + 1))
  local pass rc
  for pass in 0 1 2 3; do
    if [ "$pass" -eq 0 ]; then
      echo "================ [step $STEP/$TOTAL] $* -- $(date '+%H:%M:%S') ================" | tee -a "$LOG"
    else
      echo "---------------- [step $STEP/$TOTAL] retry pass $pass/3 over the failed rows -- $(date '+%H:%M:%S') ----------------" | tee -a "$LOG"
    fi
    set +e
    .venv/bin/python -m epai.ai_detection.rebuttal.capture --model "$MODEL" --yes "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
      return 0
    elif [ "$rc" -ne 3 ]; then
      echo "!!!! [step $STEP/$TOTAL] crashed (exit $rc) -- stopping. Re-run the script to resume." | tee -a "$LOG"
      exit "$rc"
    elif [ "$pass" -lt 3 ]; then
      echo "     some rows failed; waiting ${PAUSES[$pass]} s before retry pass $((pass + 1))" | tee -a "$LOG"
      sleep "${PAUSES[$pass]}"
    fi
  done
  echo "!!!! [step $STEP/$TOTAL] rows still failing after 3 retry passes -- moving on; re-run the script later to retry them" | tee -a "$LOG"
}
