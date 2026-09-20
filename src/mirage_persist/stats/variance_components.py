"""Variance components: sigma_tau (between-checkpoint) and within-checkpoint variance.

CV-0(d) fits these per outcome and re-solves the power table (design 5.0), retiring
the sigma_tau=0.10 assumption. REML (statsmodels MixedLM) is primary; a one-way ANOVA
method-of-moments estimator is the fallback when REML does not converge (small pilots).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class VarianceComponents:
    outcome: str
    sigma_tau: float  # between-checkpoint SD of the outcome mean
    within_var: float  # within-checkpoint (residual) variance
    n_checkpoints: int
    n_obs: int
    method: str  # "reml" | "mom"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "outcome": self.outcome,
            "sigma_tau": round(self.sigma_tau, 6),
            "sigma_tau2": round(self.sigma_tau**2, 6),
            "within_var": round(self.within_var, 6),
            "n_checkpoints": self.n_checkpoints,
            "n_obs": self.n_obs,
            "method": self.method,
        }


def _mom_components(groups: Sequence[np.ndarray]) -> tuple[float, float]:
    """One-way random-effects ANOVA method-of-moments -> (sigma_tau^2, within_var)."""
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    ns = np.array([g.size for g in groups], dtype=float)
    N = ns.sum()
    if k < 2 or N <= k:
        # Not enough structure to separate components; attribute all to within.
        pooled = np.concatenate(groups) if groups else np.array([0.0])
        return 0.0, float(pooled.var(ddof=1)) if pooled.size > 1 else 0.0
    means = np.array([g.mean() for g in groups])
    grand = np.concatenate(groups).mean()
    ss_between = float(np.sum(ns * (means - grand) ** 2))
    ss_within = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups]))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    n0 = (N - np.sum(ns**2) / N) / (k - 1)
    sigma_tau2 = max(0.0, (ms_between - ms_within) / n0) if n0 > 0 else 0.0
    return sigma_tau2, ms_within


def fit_variance_components(
    outcome: str,
    values_by_checkpoint: Mapping[str, Sequence[float]],
    *,
    method: str = "reml",
) -> VarianceComponents:
    """Fit sigma_tau and within-checkpoint variance for one outcome."""
    groups = [np.asarray([x for x in v if x is not None and not np.isnan(x)], dtype=float) for v in values_by_checkpoint.values()]
    groups = [g for g in groups if g.size > 0]
    n_checkpoints = len(groups)
    n_obs = int(sum(g.size for g in groups))

    if method == "reml" and n_checkpoints >= 3 and n_obs >= n_checkpoints + 2:
        try:
            import pandas as pd
            import statsmodels.formula.api as smf

            rows = []
            for gi, g in enumerate(groups):
                for x in g:
                    rows.append({"y": float(x), "cp": f"c{gi}"})
            df = pd.DataFrame(rows)
            if df["y"].nunique() > 1:
                model = smf.mixedlm("y ~ 1", df, groups=df["cp"])
                res = model.fit(reml=True, method="lbfgs", disp=False)
                group_var = float(np.asarray(res.cov_re).ravel()[0])
                within = float(res.scale)
                return VarianceComponents(
                    outcome, float(np.sqrt(max(0.0, group_var))), within, n_checkpoints, n_obs, "reml"
                )
        except Exception:
            pass  # fall through to MoM

    sigma_tau2, within = _mom_components(groups)
    return VarianceComponents(outcome, float(np.sqrt(sigma_tau2)), within, n_checkpoints, n_obs, "mom")


__all__ = ["VarianceComponents", "fit_variance_components"]
