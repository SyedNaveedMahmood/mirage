"""Power solver (reproduces the published table), twin-split null, variance, bootstrap."""

from __future__ import annotations

import numpy as np

from mirage_persist.stats.bootstrap import mean_ci
from mirage_persist.stats.power import power_table, var_d_c
from mirage_persist.stats.twin_split import summarize_null, twin_split_null_js, twin_split_null_scalar
from mirage_persist.stats.variance_components import fit_variance_components

# Published table (design 5.0), sigma_tau=0.10: (Var, n_sup@.10, n_sup@.075, n_tost@.075, n_tost@.05)
_EXPECTED = {
    8: (0.0725, 57, 102, 80, 179),
    12: (0.0517, 41, 73, 57, 128),
    16: (0.0413, 33, 58, 46, 102),
    24: (0.0308, 25, 44, 34, 76),
}


def test_var_d_c_matches_published():
    for K, (var, *_rest) in _EXPECTED.items():
        assert abs(var_d_c(0.10, K) - var) < 1e-3


def test_power_table_reproduces_published_within_rounding():
    rows = power_table(0.10, [8, 12, 16, 24], sup_deltas=[0.10, 0.075], tost_deltas=[0.075, 0.05])
    for r in rows:
        var, ns10, ns075, nt075, nt05 = _EXPECTED[r.K]
        assert abs(r.var_d_c - var) < 1e-3
        # sample sizes match the published table up to a +-1 rounding convention
        assert abs(r.n_superiority[0.10] - ns10) <= 1
        assert abs(r.n_superiority[0.075] - ns075) <= 1
        assert abs(r.n_equivalence[0.075] - nt075) <= 1
        assert abs(r.n_equivalence[0.05] - nt05) <= 1


def test_twin_split_null_zero_for_identical():
    assert twin_split_null_scalar([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0
    # same empirical categorical distribution (a=2, b=2 on both sides) -> JS = 0
    assert twin_split_null_js(["a", "b", "a", "b"], ["b", "a", "b", "a"], ["a", "b"]) == 0.0


def test_twin_split_null_js_positive_for_different():
    # different distributions (a-heavy vs b-heavy) -> JS > 0
    assert twin_split_null_js(["a", "a", "a"], ["b", "b", "b"], ["a", "b"]) > 0.0


def test_twin_split_null_positive_for_different():
    assert twin_split_null_scalar([1, 1, 1, 1], [0, 0, 0, 0]) == 1.0


def test_null_summary_percentiles():
    s = summarize_null("Y_branch_5", [0.0, 0.02, 0.04, 0.06, 0.08])
    assert s.n_checkpoints == 5
    assert 0.0 <= s.mean <= 0.08
    assert s.maximum == 0.08


def test_variance_components_recover_between_group_variation():
    rng = np.random.default_rng(0)
    # checkpoints with clearly different means -> sigma_tau > 0
    data = {f"c{i}": list(np.clip(rng.normal(0.2 + 0.2 * (i % 4), 0.05, 16), 0, 1)) for i in range(24)}
    vc = fit_variance_components("Y", data, method="reml")
    assert vc.sigma_tau > 0.0
    assert vc.n_checkpoints == 24


def test_mean_ci_contains_point():
    point, lo, hi = mean_ci([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], n_boot=2000)
    assert lo <= point <= hi
