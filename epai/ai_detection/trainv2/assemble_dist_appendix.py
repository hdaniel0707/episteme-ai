r"""Stack the per-backbone calibration PNGs (already produced by the
``calibrate_*_baseline.py --corpora all`` scripts) into one composite figure
per detector family, for the paper's calibration appendix.

Run AFTER the four calibration scripts (needs their PNGs on disk):

    uv run python epai/ai_detection/trainv2/calibrate_binox_baseline.py --corpora all
    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all
    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py --corpora all
    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py --corpora all --cols detector_score_radar

then:

    uv run python epai/ai_detection/trainv2/assemble_dist_appendix.py

Each source PNG is already a row of per-corpus panels (gb_orig / mg_gb /
mg_sci / mg_aisci / all); this script only stacks rows -- one per LLM
backbone -- it does not re-touch the histograms or thresholds. Row labels
reuse ``detector_names.display_detector`` so they match the names used
elsewhere in the paper.

Writes a PNG and a PDF to results/trainv2/baseline_calibration_appendix/ for
each family, named ``dist_<family>``:
    dist_binoculars    5 rows -- falcon-7b, falcon3-7b, llama-3.1-8b, mistral-v03, qwen25-7b
    dist_fastdetectgpt the same 5 backbones
    dist_radar         1 row
    dist_judge         5 rows -- the 5 main-prompt judges
    dist_logjudge      2 rows -- gemma-4-31b / gpt-5.6-luna, windowed-logprob readout
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from epai.ai_detection.trainv2.detector_names import display_detector

BASE_DIR = Path(__file__).resolve().parents[3]
CALIB_DIR = BASE_DIR / "results" / "trainv2" / "baseline_calibration"
OUT_DIR = BASE_DIR / "results" / "trainv2" / "baseline_calibration_appendix"

BINOX_DIR = CALIB_DIR / "calibrate_binox_baseline"
FASTDETECT_DIR = CALIB_DIR / "calibrate_fastdetect_baseline"
DETECTOR_DIR = CALIB_DIR / "calibrate_detector_baseline"
JUDGE_DIR = CALIB_DIR / "calibrate_judge_baseline"

DPI = 150


def _label(raw_key: str) -> str:
    """``binox/llama31-8b`` -> ``llama-3.1-8b`` (drop the family prefix)."""
    return display_detector(raw_key).split("/", 1)[-1]


#: family name -> [(png path, raw detector-names.py key), ...], in row order.
FAMILIES: dict[str, list[tuple[Path, str]]] = {
    "binoculars": [
        (BINOX_DIR / "binoculars_score.png", "binox/falcon-7b"),
        (BINOX_DIR / "binoculars_score_falcon3_7b.png", "binox/falcon3-7b"),
        (BINOX_DIR / "binoculars_score_llama31_8b.png", "binox/llama31-8b"),
        (BINOX_DIR / "binoculars_score_mistral_v03.png", "binox/mistral-v03"),
        (BINOX_DIR / "binoculars_score_qwen25_7b.png", "binox/qwen25-7b"),
    ],
    "fastdetectgpt": [
        (FASTDETECT_DIR / "fastdetect_crit_falcon_7b.png", "fastdetect/falcon-7b"),
        (FASTDETECT_DIR / "fastdetect_crit_falcon3_7b.png", "fastdetect/falcon3-7b"),
        (FASTDETECT_DIR / "fastdetect_crit_llama31_8b.png", "fastdetect/llama31-8b"),
        (FASTDETECT_DIR / "fastdetect_crit_mistral_v03.png", "fastdetect/mistral-v03"),
        (FASTDETECT_DIR / "fastdetect_crit_qwen25_7b.png", "fastdetect/qwen25-7b"),
    ],
    "radar": [
        (DETECTOR_DIR / "detector_score_radar.png", "detector/radar"),
    ],
    "judge": [
        (JUDGE_DIR / "llm_judge_score_gemma4_31b_v3.png", "judge/gemma4_31b"),
        (JUDGE_DIR / "llm_judge_score_gpt56luna_v3.png", "judge/gpt56luna"),
        (JUDGE_DIR / "llm_judge_score_gptoss_120b_v3.png", "judge/gptoss_120b"),
        (JUDGE_DIR / "llm_judge_score_llama33_70b_v3.png", "judge/llama33_70b"),
        (JUDGE_DIR / "llm_judge_score_qwen3_14b_v3.png", "judge/qwen3_14b"),
    ],
    "logjudge": [
        (JUDGE_DIR / "llm_judge_score_gemma4_31b_v3_wlp.png", "judge/gemma4_31b_wlp"),
        (JUDGE_DIR / "llm_judge_score_gpt56luna_v3_wlp.png", "judge/gpt56luna_wlp"),
    ],
}


def _assemble(name: str, rows: list[tuple[Path, str]]) -> None:
    missing = [p for p, _ in rows if not p.exists()]
    if missing:
        raise SystemExit(f"{name}: missing source PNG(s), run the calibration "
                          f"scripts first: {missing}")

    imgs = [mpimg.imread(p) for p, _ in rows]
    heights = [im.shape[0] for im in imgs]
    width = imgs[0].shape[1]
    fig_w, fig_h = width / DPI, sum(heights) / DPI

    fig, axes = plt.subplots(
        len(imgs), 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": heights}, squeeze=False,
    )
    for ax, img, (_, raw_key) in zip(axes[:, 0], imgs, rows):
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        ax.text(-0.01, 0.5, _label(raw_key), transform=ax.transAxes,
                 fontsize=13, fontweight="bold", va="center", ha="right",
                 rotation=90)
    fig.subplots_adjust(left=0.045, right=0.998, top=0.998, bottom=0.002,
                         hspace=0.015)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        out = OUT_DIR / f"dist_{name}{suffix}"
        fig.savefig(out, dpi=DPI)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    for name, rows in FAMILIES.items():
        _assemble(name, rows)
