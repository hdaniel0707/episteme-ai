#!/usr/bin/env bash
# Progress of every planned rebuttal run, from the sidecars on disk. No calls.
#   bash epai/ai_detection/rebuttal/status.sh
cd "$(dirname "$0")/../../.." && exec .venv/bin/python -m epai.ai_detection.rebuttal.status
