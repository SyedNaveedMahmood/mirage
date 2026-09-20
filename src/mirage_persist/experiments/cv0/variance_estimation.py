"""CV-0(d): variance components -> re-solved power table (design 5.0).

Consumes the per-checkpoint outcome samples produced by the determinism audit, fits
sigma_tau and the within-checkpoint variance per outcome (REML with a MoM fallback),
and re-solves the K x delta sample-size table with the measured sigma_tau -- retiring
the sigma_tau = 0.10 assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage_persist.config.schema import CV0Config
from mirage_persist.stats.power import PowerRow, power_table
from mirage_persist.stats.variance_components import VarianceComponents, fit_variance_components


@dataclass
class VarianceResult:
    components: dict[str, VarianceComponents]
    primary_outcome: str
    sigma_tau_used: float
    power_rows: list[PowerRow]

    def to_dict(self) -> dict[str, object]:
        return {
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "primary_outcome": self.primary_outcome,
            "sigma_tau_used": round(self.sigma_tau_used, 6),
            "power_table": [r.to_row() for r in self.power_rows],
        }


def run_variance_estimation(
    config: CV0Config,
    outcome_samples: dict[str, dict[str, list[float]]],
) -> VarianceResult:
    ve = config.variance_estimation
    components: dict[str, VarianceComponents] = {}
    for name in ve.outcomes:
        samples = outcome_samples.get(name, {})
        components[name] = fit_variance_components(name, samples, method=ve.method)

    primary = ve.outcomes[0] if ve.outcomes else "Y_branch_5"
    sigma_tau = components[primary].sigma_tau if primary in components else 0.10
    # If the pilot degenerates to zero variance (e.g. mock: identical continuations),
    # fall back to the design assumption so the re-solved table stays interpretable.
    sigma_tau_used = sigma_tau if sigma_tau > 0 else 0.10
    rows = power_table(sigma_tau_used, ve.power_table_K, ve.power_table_delta)
    return VarianceResult(
        components=components,
        primary_outcome=primary,
        sigma_tau_used=sigma_tau_used,
        power_rows=rows,
    )


__all__ = ["run_variance_estimation", "VarianceResult"]
