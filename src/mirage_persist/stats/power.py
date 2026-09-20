"""The shared power model (design 5.0).

For a checkpoint-paired binary outcome with K continuations per arm:

    Var(d_c) = sigma_tau^2 + [p_A(1-p_A) + p_B(1-p_B)] / K
    n_superiority(delta) = 7.85 * Var(d_c) / delta^2      (two-sided alpha=.05, power .80)
    n_equivalence(delta) = 6.18 * Var(d_c) / delta^2      (TOST, alpha=.05 one-sided, power .80)

``power_table`` reproduces the published K x delta table exactly (unit-tested):
e.g. sigma_tau=0.10, K=16 -> Var=0.04125; n_sup(0.10)=33; n_equiv(0.05)=102.
CV-0(d) re-solves this table with the measured sigma_tau.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_C_SUP = 7.85  # (z_{1-a/2}+z_{1-b})^2 with a=.05,b=.20 ~= (1.96+0.8416)^2
_C_EQ = 6.18  # (z_{1-a}+z_{1-b})^2 with a=.05,b=.20 ~= (1.6449+0.8416)^2


def var_d_c(sigma_tau: float, K: int, p_A: float = 0.5, p_B: float = 0.5) -> float:
    """Variance of the checkpoint-level paired difference."""
    return sigma_tau**2 + (p_A * (1.0 - p_A) + p_B * (1.0 - p_B)) / K


def n_superiority(var: float, delta: float) -> int:
    return math.ceil(_C_SUP * var / (delta**2))


def n_equivalence(var: float, delta: float) -> int:
    return math.ceil(_C_EQ * var / (delta**2))


@dataclass
class PowerRow:
    K: int
    var_d_c: float
    n_superiority: dict[float, int]  # delta -> n
    n_equivalence: dict[float, int]  # delta -> n

    def to_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {"K": self.K, "var_d_c": round(self.var_d_c, 5)}
        for d, n in self.n_superiority.items():
            row[f"n_sup_{d}"] = n
        for d, n in self.n_equivalence.items():
            row[f"n_tost_{d}"] = n
        return row


def power_table(
    sigma_tau: float,
    Ks: list[int],
    sup_deltas: list[float],
    tost_deltas: list[float] | None = None,
    *,
    p_A: float = 0.5,
    p_B: float = 0.5,
) -> list[PowerRow]:
    """Return one :class:`PowerRow` per K. ``tost_deltas`` defaults to ``sup_deltas``."""
    tost_deltas = tost_deltas if tost_deltas is not None else list(sup_deltas)
    rows: list[PowerRow] = []
    for K in Ks:
        v = var_d_c(sigma_tau, K, p_A, p_B)
        rows.append(
            PowerRow(
                K=K,
                var_d_c=v,
                n_superiority={d: n_superiority(v, d) for d in sup_deltas},
                n_equivalence={d: n_equivalence(v, d) for d in tost_deltas},
            )
        )
    return rows


__all__ = ["var_d_c", "n_superiority", "n_equivalence", "PowerRow", "power_table"]
