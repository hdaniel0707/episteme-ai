# Rerun list — `gb_orig` temporarily on `ghostbuster_orig_stand.parquet`

`splits.py`'s `gb_orig` entry currently points at `ghostbuster_orig_stand.parquet`
instead of `ghostbuster_orig.parquet`, because the server rescore of the raw
file is still running. Same schema/labels (`human`, `gpt35t:0000A`), so no
script changes are needed — every run below just picks it up via the default
`--corpora`/`--splits` sweep.

Run from the repo root.

## A. Baseline calibration
```bash
uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py      --corpora all
uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all
uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py   --corpora all
```

## D. Zero-shot performance
```bash
uv run python epai/ai_detection/trainv2/run_zeroshot_performance.py
uv run python epai/ai_detection/trainv2/run_best_zeroshot_performance.py
```

## F. Trained detection (train + LODO)
```bash
uv run python epai/ai_detection/trainv2/run_in_domain_performance.py
uv run python epai/ai_detection/trainv2/run_lodo_performance.py
```

---

Once the server rescore lands, switch `splits.py`'s `gb_orig` entry back to
`ghostbuster_orig.parquet` and rerun the same 7 lines above on the real data.
