"""Statistics: power solver, twin-split null, cluster bootstrap, variance components.

No hypothesis test is run in CV-0 (descriptive + CIs only). These estimators are the
shared statistical core reused by CV-1..CV-5.
"""

from __future__ import annotations

from mirage_persist.stats.bootstrap import cluster_bootstrap_ci, mean_ci
from mirage_persist.stats.power import PowerRow, power_table, var_d_c
from mirage_persist.stats.twin_split import NullSummary, summarize_null, twin_split_null_js, twin_split_null_scalar
from mirage_persist.stats.variance_components import VarianceComponents, fit_variance_components

__all__ = [
    "power_table",
    "PowerRow",
    "var_d_c",
    "twin_split_null_scalar",
    "twin_split_null_js",
    "summarize_null",
    "NullSummary",
    "cluster_bootstrap_ci",
    "mean_ci",
    "fit_variance_components",
    "VarianceComponents",
]
