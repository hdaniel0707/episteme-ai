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

# Load the big-feature toggles (and any user machine settings) from .env.workflow FIRST,
# so the deploy-specific overrides below still win over anything set in that file.
ENV_WF="$SCRIPT_DIR/.env.workflow"
if [ -f "\$ENV_WF" ]; then
    set -a
    # shellcheck disable=SC1090
    . "\$ENV_WF"
    set +a
fi

# Deploy-run machine overrides (these intentionally take precedence over .env.workflow).
export EPAI_DB=training
export UMAP_OPEN_BROWSER=false
export PARALLEL_JOBS=auto
export PARALLEL_LOAD_TARGET=0.90
export EMBED_BATCH_SIZE=64
# Live status file: always rewritten to hold one current line per workflow (+ the AE
# footer), each line only once, with growing bars — a clean 'watch'-able dashboard even
# for the detached (plain-file) run. \`cat\` or \`watch -n1 cat\` it any time.
export PROGRESS_STATUS_FILE="$LOG_DIR/progress.status"

is_true() { case "\$(printf '%s' "\${1:-true}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
join_csv() { local IFS=,; echo "\$*"; }

# Build the bootstrap source list from the per-source toggles.
boot_sources=()
is_true "\${ENABLE_OPENALEX:-true}" && boot_sources+=(openalex)
is_true "\${ENABLE_CROSSREF:-true}" && boot_sources+=(crossref)
is_true "\${ENABLE_ARXIV:-true}"    && boot_sources+=(arxiv)
is_true "\${ENABLE_S2:-true}"       && boot_sources+=(s2)

# Build the extend source list ("copy" always runs when extend is enabled — but the code
# drops it automatically when config.yaml fetch_from_scratch: true, for an OpenAlex-only DB).
ext_sources=(copy)
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
    echo "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db (sources: \$(join_csv "\${ext_sources[@]}")) ====="
    uv run python -m workflow.src.main_extend_db extend --sources "\$(join_csv "\${ext_sources[@]}")"
else
    echo "===== [2/4] database extension SKIPPED (ENABLE_EXTEND_DB=false) ====="
fi

if is_true "\${ENABLE_ANALYSE:-true}"; then
    echo "===== [3/4] dataset statistics -> workflow/data/analysis/ ====="
    uv run python -m workflow.src.main_analyse_dataset --db extended
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