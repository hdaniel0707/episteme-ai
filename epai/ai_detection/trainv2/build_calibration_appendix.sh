#!/usr/bin/env bash
# Rebuilds the calibration-appendix figures end to end: calibrate all four
# detector families (writing the per-backbone PNGs), then stack them into
# the 5 composite PNG+PDF figures.
#
#   ./epai/ai_detection/trainv2/build_calibration_appendix.sh
#
# Run from the repo root (or anywhere -- paths below are relative to this
# script's location).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py --corpora all --cols detector_score_radar
uv run python epai/ai_detection/trainv2/assemble_dist_appendix.py
