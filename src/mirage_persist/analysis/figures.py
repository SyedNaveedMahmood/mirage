"""Figures. Figure S1: the twin-split null distributions (the epsilon floor)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# A small, colour-blind-safe categorical palette (Okabe-Ito subset).
_BAR = "#4C78A8"
_MEAN = "#E45756"
_P95 = "#F58518"
_THRESH = "#54A24B"


def figure_s1(
    per_checkpoint_nulls: dict[str, list[float]],
    path: str | Path,
    *,
    branch_mean_threshold: float | None = None,
    branch_p95_threshold: float | None = None,
    branch_key: str | None = None,
    title: str = "Figure S1 - Twin-split null distributions (epsilon floor)",
) -> Path:
    """One histogram per outcome; marks mean, p95, and the branch gate thresholds."""
    outcomes = [k for k, v in per_checkpoint_nulls.items() if any(not np.isnan(x) for x in v)]
    if not outcomes:
        outcomes = list(per_checkpoint_nulls.keys())
    n = len(outcomes)
    ncols = min(3, n) if n else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for i, name in enumerate(outcomes):
        ax = axes[i // ncols][i % ncols]
        vals = np.asarray([x for x in per_checkpoint_nulls[name] if not np.isnan(x)], dtype=float)
        if vals.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=10)
            continue
        ax.hist(vals, bins=min(20, max(5, vals.size)), color=_BAR, alpha=0.85, edgecolor="white")
        mean, p95 = float(vals.mean()), float(np.percentile(vals, 95))
        ax.axvline(mean, color=_MEAN, lw=1.8, label=f"mean={mean:.3f}")
        ax.axvline(p95, color=_P95, lw=1.4, ls="--", label=f"p95={p95:.3f}")
        if branch_key is not None and name == branch_key:
            if branch_mean_threshold is not None:
                ax.axvline(branch_mean_threshold, color=_THRESH, lw=1.2, ls=":", label=f"mean gate={branch_mean_threshold}")
            if branch_p95_threshold is not None:
                ax.axvline(branch_p95_threshold, color=_THRESH, lw=1.0, ls="-.", label=f"p95 gate={branch_p95_threshold}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("twin-split null")
        ax.set_ylabel("checkpoints")
        ax.legend(fontsize=7, frameon=False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


__all__ = ["figure_s1"]
