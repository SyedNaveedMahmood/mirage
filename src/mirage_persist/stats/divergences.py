"""Distributional divergences (diagnostic outcomes D^tool / D^dest / D^act).

Unbiased MMD^2 (U-statistic) over tool-call sequences with a normalized-edit-distance
kernel, and a twin-split-corrected JS wrapper. No divergence is a CV-0 gate; these are
diagnostic and reported with the twin-split correction so finite-K bias never
manufactures an effect.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalized_edit_distance(a: Sequence[object], b: Sequence[object]) -> float:
    """Levenshtein distance between two token sequences, normalized to [0, 1]."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def edit_distance_kernel(a: Sequence[object], b: Sequence[object]) -> float:
    """RBF-style kernel on normalized edit distance: k(a,b) = 1 - d(a,b)."""
    return 1.0 - normalized_edit_distance(a, b)


def _kernel_matrix(xs: Sequence[Sequence[object]], ys: Sequence[Sequence[object]]) -> np.ndarray:
    m = np.empty((len(xs), len(ys)), dtype=float)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            m[i, j] = edit_distance_kernel(x, y)
    return m


def mmd2_unbiased(xs: Sequence[Sequence[object]], ys: Sequence[Sequence[object]]) -> float:
    """Unbiased MMD^2 U-statistic between two samples of sequences.

    Returns a value that is ~0 (possibly slightly negative) when the two samples come
    from the same distribution -- which is why it is preferred over the biased estimator
    for small K (design C4).
    """
    n, m = len(xs), len(ys)
    if n < 2 or m < 2:
        return float("nan")
    kxx = _kernel_matrix(xs, xs)
    kyy = _kernel_matrix(ys, ys)
    kxy = _kernel_matrix(xs, ys)
    sum_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    sum_xy = kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


__all__ = ["normalized_edit_distance", "edit_distance_kernel", "mmd2_unbiased"]
