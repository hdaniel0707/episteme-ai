"""Figure 3 -- AUROC vs the deployable operating point, one marker per detector.

The point the AUROC table cannot make on its own: a detector can rank documents
well (high AUROC) and still be useless at the 1%-false-positive operating point
anyone would actually deploy at. Judges sit high-left, trained heads high-right.

Each detector is drawn as a filled marker at the TRANSPORTED TPR@FPR (threshold
fit on the source's calibration slice -- the deployable number) and an open
marker at the ORACLE TPR@FPR (the target's own ROC-optimal threshold), joined by
a stem. In-domain the stem is short; under transfer it opens up, and that gap is
the cost of a fixed threshold under domain shift.

``plot_operating_point`` takes one or more panels; the caller builds the panel
data from whatever comparison payload it has:

  * :func:`id_panels`  -- a clf_id ``corpus_comparison.json`` (one panel)
  * :func:`cd_panels`  -- a clf_cd ``suite_comparison.json`` / ``variant_comparison
    .json`` (two panels: in-domain, transfer)
"""

from __future__ import annotations

from pathlib import Path

# Marker shape carries the family (the plan's "marker shape by family"); hue only
# separates trained from zero-shot, the same split every other figure in the
# repo draws.
FAMILY_MARKER: dict[str, str] = {
    "trained": "o", "hybrid": "D",
    "binoculars": "^", "fastdetect": "v", "llm_judge": "s", "detector": "*",
}
FAMILY_LABEL: dict[str, str] = {
    "trained": "trained head", "hybrid": "hybrid (feat + cues)",
    "binoculars": "Binoculars", "fastdetect": "Fast-DetectGPT",
    "llm_judge": "LLM judge", "detector": "pretrained detector",
}
_TRAINED_COLOR = "#2e8b57"
_FAMILY_COLOR: dict[str, str] = {
    "trained": _TRAINED_COLOR, "hybrid": _TRAINED_COLOR,
    "binoculars": "#3f7fb5", "fastdetect": "#1b9e77",
    "llm_judge": "#6a51a3", "detector": "#b5651d",
}


def variant_family(variant: str, label: str = "") -> str:
    """The detector family of a comparison row, from its variant id."""
    v = (variant or "").lower()
    for key in ("binoculars", "fastdetect", "llm_judge"):
        if v == key.split("_", 1)[0] or v.startswith(key):
            return key
    if v == "detector" or v.startswith("detector_"):
        return "detector"
    if "cuevec" in v and ("fusion" in v or "gated" in v):
        return "hybrid"
    return "trained"


def _point(row: dict, auroc_key: str, deploy_key: str, oracle_key: str) -> dict | None:
    a = row.get(auroc_key)
    d = row.get(deploy_key)
    o = row.get(oracle_key)
    if a is None or (d is None and o is None):
        return None
    return {
        "label": row.get("label") or row.get("variant") or "?",
        "family": variant_family(row.get("variant", ""), row.get("label", "")),
        "baseline": bool(row.get("baseline")),
        "auroc": float(a),
        "deploy": None if d is None else float(d),
        "oracle": None if o is None else float(o),
    }


def id_panels(comparison: dict, level: str = "0.01") -> list[tuple[str, list[dict]]]:
    rows = comparison.get("variants", [])
    pts = [p for p in (
        _point(r, "auroc_mean", f"tpr_at_srcfpr_{level}_mean", f"tpr_at_fpr_{level}_mean")
        for r in rows) if p]
    return [("in-domain", pts)]


def cd_panels(comparison: dict, level: str = "0.01") -> list[tuple[str, list[dict]]]:
    rows = comparison.get("variants", [])
    out = []
    for prefix, title in (("in_domain_", "in-domain (diagonal)"),
                          ("transfer_", "transfer (unseen target)")):
        pts = [p for p in (
            _point(r, f"{prefix}auroc_mean",
                   f"{prefix}tpr_at_srcfpr_{level}_mean",
                   f"{prefix}tpr_at_fpr_{level}_mean")
            for r in rows) if p]
        if pts:
            out.append((title, pts))
    return out


def plot_operating_point(panels: list[tuple[str, list[dict]]], out_path: Path, *,
                         title: str, subtitle: str = "", level: str = "0.01") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [(t, p) for t, p in panels if p]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.6),
                             squeeze=False)
    axes = axes[0]

    fams: set[str] = set()
    for ax, (ptitle, pts) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "--", lw=1, color="#bbbbbb", zorder=1)
        for p in pts:
            fams.add(p["family"])
            c = _FAMILY_COLOR.get(p["family"], "#555")
            m = FAMILY_MARKER.get(p["family"], "o")
            lo = p["deploy"] if p["deploy"] is not None else p["oracle"]
            hi = p["oracle"] if p["oracle"] is not None else p["deploy"]
            if lo != hi:
                ax.plot([p["auroc"], p["auroc"]], [min(lo, hi), max(lo, hi)],
                        "-", lw=1, color=c, alpha=0.45, zorder=2)
            if p["oracle"] is not None:
                ax.scatter([p["auroc"]], [p["oracle"]], s=60, marker=m,
                           facecolors="none", edgecolors=c, lw=1.3, zorder=3)
            if p["deploy"] is not None:
                ax.scatter([p["auroc"]], [p["deploy"]], s=60, marker=m,
                           color=c, zorder=4)
        ax.set_title(ptitle)
        ax.set_xlabel("AUROC (machine = positive)")
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, lw=0.4, alpha=0.4)
    axes[0].set_ylabel(f"TPR at {level} FPR")

    handles = [Line2D([], [], marker=FAMILY_MARKER[f], ls="", color=_FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f]) for f in FAMILY_MARKER if f in fams]
    handles += [
        Line2D([], [], marker="o", ls="", color="#555", label="transported (deployable)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#555", label="oracle (target-calibrated)"),
    ]
    # Upper-left: nothing sits at low AUROC / high TPR, so the legend never
    # covers a marker there.
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.92)

    sup = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle else 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
