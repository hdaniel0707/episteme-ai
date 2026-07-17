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
export EPAI_DB=training
export UMAP_OPEN_BROWSER=false
export PARALLEL_JOBS=auto
export PARALLEL_LOAD_TARGET=0.90
export EMBED_BATCH_SIZE=64

echo "===== [1/4] bootstrap: fresh multi-source fetch -> raw.db ====="
echo "      (resume-safe: re-running this step alone continues where it left off)"
uv run python -m workflow.src.main_extend_db bootstrap --target-rows ${TARGET_ROWS}

echo "===== [2/4] extend: raw.db + legacy DBs -> extended.db -> training.db ====="
uv run python -m workflow.src.main_extend_db extend

echo "===== [3/4] dataset statistics -> workflow/data/analysis/ ====="
uv run python -m workflow.src.main_analyse_dataset --db extended

echo "===== [4/4] run_all_workflows: the training + benchmark queue ====="
uv run python -m workflow.run_all_workflows

echo ""
echo "===== PIPELINE COMPLETE \$(date) ====="
echo "results -> $REPO_ROOT/workflow/results/  (see results/summary/ for cross-run figures)"
INNEREOF
chmod +x "$INNER"
ln -sf "$(basename "$RUN_LOG")" "$LOG_DIR/latest.log"

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" "bash '$INNER' 2>&1 | tee '$RUN_LOG'"
    log "launched in tmux session '$SESSION' — survives lost connections."
    echo "        Reattach any time:   tmux attach -t $SESSION"
    echo "        Detach again:        Ctrl-b then d"
else
    log "tmux not found — falling back to nohup+setsid (also survives lost connections)."
    setsid nohup bash "$INNER" > "$RUN_LOG" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/pipeline.pid"
    disown
    log "launched detached, PID $(cat "$LOG_DIR/pipeline.pid")"
fi

echo ""
echo "[deploy] Monitor anytime with:   tail -f $LOG_DIR/latest.log"
echo "[deploy] You can safely close this terminal / lose the SSH connection now."