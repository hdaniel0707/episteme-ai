#!/usr/bin/env bash
# workflow/stop_deploy_and_run.sh — stop the detached pipeline and clear best-effort
# RAM / VRAM usage by terminating the pipeline process tree and any GPU consumers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SESSION="epai_pipeline"
PID_FILE="$LOG_DIR/pipeline.pid"

stop_tree() {
    local pid="$1"
    local children=()
    if command -v pgrep >/dev/null 2>&1; then
        mapfile -t children < <(pgrep -P "$pid" 2>/dev/null || true)
        for child in "${children[@]}"; do
            stop_tree "$child"
        done
    fi
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true
}

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "[stop] killed tmux session $SESSION"
fi

if [ -f "$PID_FILE" ]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        stop_tree "$pid"
        echo "[stop] killed pipeline pid $pid"
    fi
    rm -f "$PID_FILE"
fi

pkill -TERM -f 'workflow\.(deploy_and_run|run_all_workflows|run_pipeline)|workflow\.src\.(main_extend_db|main_analyse_dataset)' 2>/dev/null || true
sleep 1
pkill -KILL -f 'workflow\.(deploy_and_run|run_all_workflows|run_pipeline)|workflow\.src\.(main_extend_db|main_analyse_dataset)' 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t gpu_pids < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | tr -d '[:space:]' \
            | grep -E '^[0-9]+$' \
            | sort -u || true
    )
    for pid in "${gpu_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${gpu_pids[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
fi

sync
if [ "$(id -u)" -eq 0 ] && [ -w /proc/sys/vm/drop_caches ]; then
    echo 3 > /proc/sys/vm/drop_caches
fi

echo "[stop] pipeline stopped; RAM/VRAM should now be released by the OS."