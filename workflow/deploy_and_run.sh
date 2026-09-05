#!/usr/bin/env bash
# workflow/deploy_and_run.sh — detached-by-default full pipeline launcher.
# Run this from a cloned checkout. It keeps the data-collection + training job
# running even if the SSH/JupyterLab session disappears.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SESSION="epai_pipeline"
TARGET_ROWS="${TARGET_ROWS:-100000}"

log() { printf '\n[deploy] %s\n' "$*"; }

# The launcher itself is configurable from .env.workflow too. Read only the keys it needs
# (the inner script sources the whole file); the values here are fallbacks for absent keys.
deploy_cfg() {   # deploy_cfg KEY FALLBACK -> value from .env.workflow, else FALLBACK
    local v=""
    if [ -f "$SCRIPT_DIR/.env.workflow" ]; then
        v="$(sed -n "s/^[[:space:]]*$1[[:space:]]*=//p" "$SCRIPT_DIR/.env.workflow" \
             | head -1 | cut -d'#' -f1 | xargs || true)"
    fi
    if [ -z "$v" ]; then printf '%s' "$2"; else printf '%s' "$v"; fi
}
deploy_on() {    # deploy_on KEY [FALLBACK] -> 0 when the key is true-ish
    case "$(printf '%s' "$(deploy_cfg "$1" "${2:-true}")" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;; *) return 1 ;;
    esac
}

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv (package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv sync — installing the project's pinned dependencies"
uv sync --frozen

if [ ! -f .env ]; then
    cp .env_sample .env
    log "created .env from .env_sample."
    echo "        >>> Only needed if you enable keyed features (llm_judge baseline,"
    echo "        >>> naip_local weights download, MIRAI generated-paper scoring)."
    echo "        >>> The default pipeline (bootstrap/extend/analyse/run_all_workflows"
    echo "        >>> with specter2_knn + venue_ais baselines) needs NO keys at all."
    echo "        >>> Edit $REPO_ROOT/.env now if you want them, then re-run this script."
else
    log ".env already present — leaving it untouched"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/pipeline_${TS}.log"
RUN_ERR="$LOG_DIR/pipeline_${TS}.err"
INNER="$LOG_DIR/pipeline_${TS}.sh"

cat > "$INNER" <<INNEREOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
export PATH="\$HOME/.local/bin:\$PATH"

# Load the big-feature toggles (and any user machine settings) from .env.workflow FIRST;
# the exports below only FILL IN keys that file does not set, so it always wins.
ENV_WF="$SCRIPT_DIR/.env.workflow"
if [ -f "\$ENV_WF" ]; then
    set -a
    # shellcheck disable=SC1090
    . "\$ENV_WF"
    set +a
fi

# Machine knobs — .env.workflow (sourced above) WINS; these are only the fallbacks used
# when it does not set the key, so nothing here is hardcoded over your settings.
# The progress bars/ticks are UTF-8 glyphs; force UTF-8 I/O so a C/POSIX locale cannot
# turn them into mojibake in the log or the live status file.
export PYTHONIOENCODING="\${PYTHONIOENCODING:-utf-8}"
# Unbuffered: a detached run redirects stdout to a file and Python block-buffers that, so
# without this the log stays empty and a stalled run looks identical to a working one.
export PYTHONUNBUFFERED="\${PYTHONUNBUFFERED:-1}"
export EPAI_DB="\${EPAI_DB:-training}"
export UMAP_OPEN_BROWSER="\${UMAP_OPEN_BROWSER:-false}"
export PARALLEL_JOBS="\${PARALLEL_JOBS:-auto}"
export PARALLEL_LOAD_TARGET="\${PARALLEL_LOAD_TARGET:-0.90}"
export EMBED_BATCH_SIZE="\${EMBED_BATCH_SIZE:-64}"
# Live status file: always rewritten to hold one current line per workflow (+ the AE
# footer), each line only once, with growing bars — a clean 'watch'-able dashboard even
# for the detached (plain-file) run. \`cat\` or \`watch -n1 cat\` it any time.
export PROGRESS_STATUS_FILE="$LOG_DIR/progress.status"
# Drop the previous session's status: until training starts the dashboard must show
# nothing rather than the last run's finished models.
rm -f "$LOG_DIR/progress.status"

is_true() { case "\$(printf '%s' "\${1:-true}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
join_csv() { local IFS=,; echo "\$*"; }

# Build the bootstrap source list from the per-source toggles.
boot_sources=()
is_true "\${ENABLE_OPENALEX:-true}" && boot_sources+=(openalex)
is_true "\${ENABLE_CROSSREF:-true}" && boot_sources+=(crossref)
is_true "\${ENABLE_ARXIV:-true}"    && boot_sources+=(arxiv)
is_true "\${ENABLE_S2:-true}"       && boot_sources+=(s2)

# Build the extend source list. ENABLE_COPY=false keeps raw.db + the legacy store DBs OUT
# of extended.db, so extend only re-derives training.db and writes to no collected data.
# (The code also drops 'copy' automatically when config.yaml fetch_from_scratch: true.)
ext_sources=()
is_true "\${ENABLE_COPY:-true}"     && ext_sources+=(copy)
is_true "\${ENABLE_PDF:-true}"      && ext_sources+=(pdf)
is_true "\${ENABLE_UEFISCDI:-true}" && ext_sources+=(uefiscdi)
is_true "\${ENABLE_OPENALEX:-true}" && ext_sources+=(openalex)
is_true "\${ENABLE_S2:-true}"       && ext_sources+=(s2)
is_true "\${ENABLE_FILL:-true}"     && ext_sources+=(fill)
# llm_fill runs when domain OR abstract auto-fill is enabled (LLM domain/abstract completion).
if is_true "\${ENABLE_LLM_FILL:-false}" || is_true "\${ENABLE_LLM_ABSTRACT_FILL:-false}"; then
    ext_sources+=(llm_fill)
fi

if is_true "\${ENABLE_BOOTSTRAP:-true}" && [ \${#boot_sources[@]} -gt 0 ]; then
    echo "===== [1/4] bootstrap: fresh multi-source fetch -> raw.db (sources: \$(join_csv "\${boot_sources[@]}")) ====="
    echo "      (resume-safe: re-running this step alone continues where it left off)"
    uv run python -m workflow.src.main_extend_db bootstrap --target-rows ${TARGET_ROWS} --sources "\$(join_csv "\${boot_sources[@]}")"
else
    echo "===== [1/4] bootstrap SKIPPED (ENABLE_BOOTSTRAP=false or no sources enabled) ====="
fi

if is_true "\${ENABLE_EXTEND_DB:-true}"; then
    # 'none' = every source off: no DB is modified, extend only re-derives training.db
    # (seeded random subset of extended.db) + report + local benchmarks.
    [ \${#ext_sources[@]} -eq 0 ] && ext_sources=(none)
    echo "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db (sources: \$(join_csv "\${ext_sources[@]}")) ====="
    uv run python -m workflow.src.main_extend_db extend --sources "\$(join_csv "\${ext_sources[@]}")"
else
    echo "===== [2/4] database extension SKIPPED (ENABLE_EXTEND_DB=false) ====="
fi

if is_true "\${ENABLE_ANALYSE:-true}"; then
    # Two suites in one run: the olive figures describe ANALYSE_DB (the whole collected
    # corpus, default extended) and the red '_train' figures describe the rows actually
    # trained on (EPAI_DB). Set them to the same DB and the two suites become identical.
    ANALYSE_DB="\${ANALYSE_DB:-extended}"
    echo "===== [3/4] dataset statistics (corpus: \$ANALYSE_DB, trained-on: \$EPAI_DB) -> workflow/data/analysis/ ====="
    uv run python -m workflow.src.main_analyse_dataset --db "\$ANALYSE_DB" --train-db "\$EPAI_DB"
else
    echo "===== [3/4] dataset statistics SKIPPED (ENABLE_ANALYSE=false) ====="
fi

if is_true "\${ENABLE_TRAINING:-true}"; then
    echo "===== [4/4] run_all_workflows: the training + benchmark queue ====="
    uv run python -m workflow.run_all_workflows
else
    echo "===== [4/4] training SKIPPED (ENABLE_TRAINING=false) ====="
fi

echo ""
echo "===== PIPELINE COMPLETE \$(date) ====="
echo "results -> $REPO_ROOT/workflow/results/  (see results/summary/ for cross-run figures)"
INNEREOF
chmod +x "$INNER"
ln -sf "$(basename "$RUN_LOG")" "$LOG_DIR/latest.log"

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    # FORCE_PROGRESS_ANSI=1 keeps the live progress lines refreshing IN PLACE inside the
    # tmux pane: without it, piping through `tee` hides the TTY from the pipeline and the
    # progress display falls back to appending a new line every second. tee still forwards
    # the escape codes to the (TTY) pane, so an attached session shows a live, single-line
    # updating display. (The .log copy keeps the raw escape codes; `tail -f` renders fine.)
    tmux new-session -d -s "$SESSION" "FORCE_PROGRESS_ANSI=1 bash '$INNER' 2>&1 | tee '$RUN_LOG'"
    log "launched in tmux session '$SESSION' — survives lost connections."
    echo "        Reattach any time:   tmux attach -t $SESSION  (progress refreshes live there)"
    echo "        Detach again:        Ctrl-b then d"
else
    log "tmux not found — falling back to nohup+setsid (also survives lost connections)."
    # Detached to a plain log file (no TTY): the progress display auto-throttles to one
    # line per real state change instead of per-second ETA spam.
    setsid nohup bash "$INNER" > "$RUN_LOG" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/pipeline.pid"
    disown
    log "launched detached, PID $(cat "$LOG_DIR/pipeline.pid")"
fi

echo ""
echo "[deploy] Full scrolling log:      tail -f $LOG_DIR/latest.log"
echo "[deploy] Live progress dashboard: watch -n1 cat $LOG_DIR/progress.status"
echo "                                  (one line per workflow, updated in place, + AE footer)"
echo "[deploy] You can safely close this terminal / lose the SSH connection now."

# Without this the launcher returned instantly and the terminal closed on top of the run.
# FOLLOW_PROGRESS=false in .env.workflow skips it; FOLLOW_DELAY_SECONDS sets the wait.
FOLLOW_DELAY="$(deploy_cfg FOLLOW_DELAY_SECONDS 5)"
if deploy_on FOLLOW_PROGRESS && [ -t 1 ]; then
    echo ""
    echo "[deploy] The live progress dashboard opens in ${FOLLOW_DELAY}s. It is only a VIEWER:"
    if command -v tmux >/dev/null 2>&1; then
        echo "         detach with Ctrl-b then d — the run keeps going in tmux session '$SESSION'."
        echo "         Rejoin later:  tmux attach -t $SESSION"
    else
        echo "         press Ctrl-C any time — the run keeps going detached (PID in logs/pipeline.pid)."
        echo "         Rejoin later:  watch -n1 cat $LOG_DIR/progress.status"
    fi
    echo "         Turn this auto-open off: FOLLOW_PROGRESS=false in workflow/.env.workflow"
    i="$FOLLOW_DELAY"
    while [ "$i" -gt 0 ]; do
        printf '\r         opening in %s s ... (Ctrl-C to stay here)   ' "$i"
        sleep 1
        i=$((i - 1))
    done
    printf '\n'
    if command -v tmux >/dev/null 2>&1; then
        tmux attach -t "$SESSION"
    elif command -v watch >/dev/null 2>&1; then
        watch -n1 cat "$LOG_DIR/progress.status"
    else
        while kill -0 "$(cat "$LOG_DIR/pipeline.pid")" 2>/dev/null; do
            clear; cat "$LOG_DIR/progress.status" 2>/dev/null || echo "(waiting for the first status tick...)"
            sleep 1
        done
    fi
fi