"""Bootstrap confidence intervals (cluster bootstrap over checkpoints).

The primary independent unit is the checkpoint (design 3.0); repeated continuations
from the same checkpoint are not independent, so intervals resample *checkpoints*.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of iid values -> (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(v.mean()), float(lo), float(hi))


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI: resample whole clusters (checkpoints) with replacement."""
    arrays = [np.asarray(c, dtype=float) for c in clusters if len(c) > 0]
    if not arrays:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    k = len(arrays)
    point = statistic(np.concatenate(arrays))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        pooled = np.concatenate([arrays[i] for i in pick])
        boots[b] = statistic(pooled)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


__all__ = ["mean_ci", "cluster_bootstrap_ci"]
