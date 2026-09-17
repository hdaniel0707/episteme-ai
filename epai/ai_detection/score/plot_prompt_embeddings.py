r"""Plot every judge-prompt candidate (plus ``v3``) in roberta-base embedding
space -- 2D and 3D PCA, each as its own interactive HTML file and its own
static PDF.

Sanity-check companion to ``select_diverse_judge_prompts.py``: uses the exact
same embedding recipe (``roberta-base``, mean-pooled last hidden state) and
the exact same worst-case-distance rule to find the winning pair, so the plot
always shows whichever pair that script would currently pick -- however many
candidate YAMLs are sitting in ``cfg/ai_detection/judge_prompts/candidates/``
at the time it's run.

Marks three things distinctly, everything else plain:
    the production prompt (``v3``, the anchor everything is measured against)
    the winning pair (the two candidates the selector would pick right now)
    every other candidate

Writes 4 files into ``--out-dir``: ``pca2d.html``, ``pca2d.pdf``,
``pca3d.html``, ``pca3d.pdf``. The PDF is a static snapshot (fixed camera
angle for the 3D one); the HTML stays hoverable/rotatable.

Usage:
    .venv/bin/python epai/ai_detection/score/plot_prompt_embeddings.py
    .venv/bin/python epai/ai_detection/score/plot_prompt_embeddings.py --metric euclidean
    .venv/bin/python epai/ai_detection/score/plot_prompt_embeddings.py \
        --candidates-dir cfg/ai_detection/judge_prompts/candidates_v2 \
        --out-dir results/trainv2/prompt_diversity_embeddings_v2 --open
"""

from __future__ import annotations

import argparse
import webbrowser
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from sklearn.decomposition import PCA

from epai.ai_detection.score.add_llm_judge_scores import PROMPTS
from epai.ai_detection.score.select_diverse_judge_prompts import (
    ANCHOR_PROMPT_ID,
    CANDIDATES_DIR,
    _distance,
    _embed,
    _load_candidates,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "prompt_diversity_embeddings"

ANCHOR_COLOR = "#111827"     # near-black diamond -- v3
WINNER_COLOR = "#e15759"     # red -- the pair the selector would pick
CANDIDATE_COLOR = "#4e79a7"  # blue -- everything else


def candidate_descriptions(dir_path: Path) -> dict[str, str]:
    """id -> the first clause of its description, for hover text."""
    out = {ANCHOR_PROMPT_ID: "Production prompt (anchor)"}
    for path in sorted(dir_path.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        out[spec["id"]] = spec["description"].strip().split(":")[0]
    return out


def find_winning_pair(ids: list[str], dist: np.ndarray) -> tuple[str, str]:
    """Same rule as select_diverse_judge_prompts.py: the pair maximising the
    trio's worst-case pairwise distance against the v3 anchor."""
    anchor = ids.index(ANCHOR_PROMPT_ID)
    others = [i for i in range(len(ids)) if i != anchor]
    best: tuple[float, int, int] | None = None
    for a, b in combinations(others, 2):
        worst = min(dist[anchor, a], dist[anchor, b], dist[a, b])
        if best is None or worst > best[0]:
            best = (worst, a, b)
    _, a, b = best
    return ids[a], ids[b]


def build_figure(
    ids: list[str],
    coords: np.ndarray,
    evr: np.ndarray,
    winner: tuple[str, str],
    descriptions: dict[str, str],
    three_d: bool,
) -> go.Figure:
    groups = {
        "candidate": [i for i, pid in enumerate(ids)
                      if pid != ANCHOR_PROMPT_ID and pid not in winner],
        "winner": [i for i, pid in enumerate(ids) if pid in winner],
        "anchor": [i for i, pid in enumerate(ids) if pid == ANCHOR_PROMPT_ID],
    }
    hover = [f"<b>{pid}</b><br>{descriptions.get(pid, '')}" for pid in ids]

    # (group, colour, marker symbol, marker size, direct-label the point)
    specs = [
        ("candidate", CANDIDATE_COLOR, "circle", 4 if three_d else 7, False),
        ("winner", WINNER_COLOR, "circle", 7 if three_d else 13, True),
        ("anchor", ANCHOR_COLOR, "diamond", 7 if three_d else 13, True),
    ]

    fig = go.Figure()
    Scatter = go.Scatter3d if three_d else go.Scatter
    for key, color, symbol, size, labelled in specs:
        idx = groups[key]
        if not idx:
            continue
        text = [ids[i] for i in idx] if labelled else None
        mode = "markers+text" if labelled else "markers"
        marker_line = dict(width=1.5, color="white") if labelled else dict(width=0)
        coord_kwargs = (
            dict(x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2]) if three_d
            else dict(x=coords[idx, 0], y=coords[idx, 1])
        )
        fig.add_trace(Scatter(
            **coord_kwargs, mode=mode,
            name=key, text=text, textposition="top center",
            hovertext=[hover[i] for i in idx], hoverinfo="text",
            marker=dict(color=color, symbol=symbol, size=size,
                        opacity=1.0 if labelled else 0.65, line=marker_line),
        ))

    title = (f"Judge-prompt candidates — {'3D' if three_d else '2D'} PCA "
             f"({len(ids)} points; 3 components capture "
             f"{evr[:3].sum() * 100:.1f}% of variance)")
    layout = dict(
        template="plotly_white", title=title,
        height=800, width=1000, legend=dict(itemsizing="constant"),
    )
    if three_d:
        layout["scene"] = dict(
            xaxis_title=f"PC1 ({evr[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({evr[1] * 100:.1f}%)",
            zaxis_title=f"PC3 ({evr[2] * 100:.1f}%)",
        )
    else:
        layout["xaxis_title"] = f"PC1 ({evr[0] * 100:.1f}%)"
        layout["yaxis_title"] = f"PC2 ({evr[1] * 100:.1f}%)"
    fig.update_layout(**layout)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                   help="Distance metric, same choice as select_diverse_judge_prompts.py.")
    p.add_argument("--candidates-dir", type=Path, default=CANDIDATES_DIR,
                   help="Directory of candidate YAMLs, e.g. a v2 re-run's own "
                   "directory. Default: the original pilot's candidates/.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory: writes pca2d.html, pca2d.pdf, "
                   "pca3d.html, pca3d.pdf here.")
    p.add_argument("--open", action="store_true",
                   help="Open both HTML plots in the default browser once written.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    candidates = _load_candidates(args.candidates_dir)
    ids = [ANCHOR_PROMPT_ID, *candidates]
    texts = [PROMPTS[ANCHOR_PROMPT_ID], *candidates.values()]
    print(f"embedding {len(texts)} prompts with roberta-base ({args.metric} distance)...")
    vecs = _embed(texts)

    n = len(ids)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = _distance(vecs[i], vecs[j], args.metric)
        dist[i, j] = dist[j, i] = d

    winner = find_winning_pair(ids, dist)
    print(f"winning pair: {winner[0]!r} + {winner[1]!r}")

    pca = PCA(n_components=3, random_state=0).fit(vecs)
    coords = pca.transform(vecs)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance: {evr.round(4).tolist()} (sum {evr.sum():.3f})")

    descriptions = candidate_descriptions(args.candidates_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for three_d, tag in ((False, "pca2d"), (True, "pca3d")):
        fig = build_figure(ids, coords, evr, winner, descriptions, three_d)

        html_out = out_dir / f"{tag}.html"
        fig.write_html(html_out, include_plotlyjs=True)
        print(f"wrote {html_out}")

        pdf_out = out_dir / f"{tag}.pdf"
        fig.write_image(pdf_out, format="pdf")
        print(f"wrote {pdf_out}")

        if args.open:
            webbrowser.open(html_out.resolve().as_uri())


if __name__ == "__main__":
    main()
