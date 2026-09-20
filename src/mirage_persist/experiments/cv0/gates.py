"""CV-0 PASS/KILL gate evaluation (design CV-0 card)."""

from __future__ import annotations

from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.base import GateResult, GateStatus
from mirage_persist.experiments.cv0.capability_screen import CapabilityResult
from mirage_persist.experiments.cv0.determinism_audit import DeterminismResult
from mirage_persist.experiments.cv0.potency_screen import PotencyResult


def evaluate_gates(
    config: CV0Config,
    det: DeterminismResult,
    cap: CapabilityResult,
    pot: PotencyResult,
) -> list[GateResult]:
    g = config.gates
    gates: list[GateResult] = []

    # -- Gate 1 + KILL: digest equality ---------------------------------- #
    rate = det.digest_equality_rate
    if rate < g.kill_digest_equality:
        gates.append(GateResult("digest_equality", GateStatus.KILL,
            f"digest equality {rate:.3f} < KILL {g.kill_digest_equality} (Substrate A determinism broken)",
            rate, g.kill_digest_equality))
    elif rate < g.digest_equality_min:
        gates.append(GateResult("digest_equality", GateStatus.FAIL,
            f"digest equality {rate:.3f} < PASS {g.digest_equality_min}", rate, g.digest_equality_min))
    else:
        gates.append(GateResult("digest_equality", GateStatus.PASS,
            f"digest equality {rate:.3f} >= {g.digest_equality_min}", rate, g.digest_equality_min))

    # -- Gate 2 + KILL: twin-split null on Y^branch_h -------------------- #
    h = config.determinism_audit.branch_horizon_for_epsilon
    key = f"Y_branch_{h}"
    summ = det.null_summaries.get(key)
    if summ is None:
        gates.append(GateResult("twin_null_branch", GateStatus.INFO,
            f"no null summary for {key}"))
    elif summ.mean > g.kill_twin_null_mean:
        gates.append(GateResult("twin_null_branch", GateStatus.KILL,
            f"{key} twin-null mean {summ.mean:.3f} > KILL {g.kill_twin_null_mean} (cannot resolve at margin)",
            summ.mean, g.kill_twin_null_mean))
    elif summ.mean > g.twin_null_branch_mean_max or summ.p95 > g.twin_null_branch_p95_max:
        gates.append(GateResult("twin_null_branch", GateStatus.FAIL,
            f"{key} twin-null mean {summ.mean:.3f}(<= {g.twin_null_branch_mean_max}) "
            f"p95 {summ.p95:.3f}(<= {g.twin_null_branch_p95_max})", summ.mean, g.twin_null_branch_mean_max))
    else:
        gates.append(GateResult("twin_null_branch", GateStatus.PASS,
            f"{key} twin-null mean {summ.mean:.3f} p95 {summ.p95:.3f} within floor",
            summ.mean, g.twin_null_branch_mean_max))

    # -- Gate 3: eligible checkpoints per cell --------------------------- #
    if cap.per_cell_eligible:
        min_cell = min(cap.per_cell_eligible.values())
        worst = min(cap.per_cell_eligible, key=cap.per_cell_eligible.get)
        status = GateStatus.PASS if min_cell >= g.eligible_per_cell_min else GateStatus.FAIL
        gates.append(GateResult("eligible_per_cell", status,
            f"min eligible/cell = {min_cell} ({worst}); need >= {g.eligible_per_cell_min}",
            float(min_cell), float(g.eligible_per_cell_min)))
    else:
        gates.append(GateResult("eligible_per_cell", GateStatus.FAIL, "no cells screened"))

    # -- Gate 4: potency (>=1 class beats BA by margin) ------------------ #
    n_pass = len(pot.classes_passing)
    status = GateStatus.PASS if n_pass >= 1 else GateStatus.FAIL
    gates.append(GateResult("potency", status,
        f"{n_pass} class(es) with diversion >= BA + {g.potency_pp_above_ba_min}: {pot.classes_passing}",
        float(n_pass), 1.0))

    # -- Gate 5 + KILL: invalid-action floor across families ------------- #
    n_fam_ok = len(cap.families_clearing_invalid_floor)
    if n_fam_ok < g.kill_min_families:
        gates.append(GateResult("capability_families", GateStatus.KILL,
            f"only {n_fam_ok} families clear invalid<= {g.invalid_action_max} floor; "
            f"< {g.kill_min_families} -> escalate to OR-6 (frontier)", float(n_fam_ok), float(g.kill_min_families)))
    elif n_fam_ok < g.min_families_pass_capability:
        gates.append(GateResult("capability_families", GateStatus.FAIL,
            f"{n_fam_ok} families clear invalid floor; need >= {g.min_families_pass_capability}",
            float(n_fam_ok), float(g.min_families_pass_capability)))
    else:
        gates.append(GateResult("capability_families", GateStatus.PASS,
            f"{n_fam_ok} families clear invalid<= {g.invalid_action_max} floor",
            float(n_fam_ok), float(g.min_families_pass_capability)))

    return gates


__all__ = ["evaluate_gates"]
