"""Twin-split null: the epsilon floor of the instrument (design 2.1, C4).

Empirical divergence between two size-K samples is positively biased; an "effect"
appears between two *identical* distributions at small K. We measure that bias
directly: from a checkpoint's never-treated continuations we take two disjoint
halves (the two twin restores) and compute their divergence. Aggregated over
checkpoints this IS the epsilon floor every later effect is compared against.

For a scalar/binary outcome Y the null is |mean(twinA) - mean(twinB)|; for a
categorical distribution (terminal node, action class) it is the JS divergence
between the two halves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def twin_split_null_scalar(a: Sequence[float], b: Sequence[float]) -> float:
    """|mean(a) - mean(b)| for two twin samples of a scalar/binary outcome."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    return float(abs(a_arr.mean() - b_arr.mean()))


def twin_split_null_random(
    values: Sequence[float], rng: np.random.Generator, n_splits: int = 50
) -> float:
    """Mean over random balanced splits of a pooled sample (when twins are not tracked)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        return float("nan")
    half = n // 2
    diffs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = v[perm[:half]], v[perm[half : 2 * half]]
        diffs.append(abs(a.mean() - b.mean()))
    return float(np.mean(diffs))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (range [0,1]) between two count/prob vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def twin_split_null_js(
    a_labels: Sequence[object], b_labels: Sequence[object], categories: Sequence[object]
) -> float:
    """JS divergence between the empirical categorical distributions of two twins."""
    cats = list(categories)
    index = {c: i for i, c in enumerate(cats)}
    pa = np.zeros(len(cats))
    pb = np.zeros(len(cats))
    for x in a_labels:
        if x in index:
            pa[index[x]] += 1
    for x in b_labels:
        if x in index:
            pb[index[x]] += 1
    return _js_divergence(pa, pb)


@dataclass
class NullSummary:
    outcome: str
    n_checkpoints: int
    mean: float
    std: float
    p95: float
    p99: float
    maximum: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "outcome": self.outcome,
            "n_checkpoints": self.n_checkpoints,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
            "max": round(self.maximum, 6),
        }


def summarize_null(outcome: str, values: Sequence[float]) -> NullSummary:
    """Aggregate per-checkpoint null values into the epsilon-floor summary."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return NullSummary(outcome, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return NullSummary(
        outcome=outcome,
        n_checkpoints=int(v.size),
        mean=float(v.mean()),
        std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
        p95=float(np.percentile(v, 95)),
        p99=float(np.percentile(v, 99)),
        maximum=float(v.max()),
    )


__all__ = [
    "twin_split_null_scalar",
    "twin_split_null_random",
    "twin_split_null_js",
    "summarize_null",
    "NullSummary",
]
