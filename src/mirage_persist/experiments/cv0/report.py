"""CV-0 orchestrator + artifact writers.

``run_cv0`` runs all four sub-experiments, evaluates the gates, and writes every
artifact into the run directory (and mirrors the canonical four into artifacts/):
    envelope.json, Figure S1, eligibility.parquet, capability_report.md,
    potency.json, variance_power.json, cv0_verdict.json.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mirage_persist.analysis.figures import figure_s1
from mirage_persist.analysis.io import write_parquet
from mirage_persist.config.schema import CV0Config
from mirage_persist.events.writer import EventWriter
from mirage_persist.experiments.base import GateResult, build_adapter, build_backends, build_scaffolds, overall_status
from mirage_persist.experiments.cv0.capability_screen import CapabilityResult, run_capability_screen
from mirage_persist.experiments.cv0.common import collect_checkpoints
from mirage_persist.experiments.cv0.determinism_audit import DeterminismResult, run_determinism_audit
from mirage_persist.experiments.cv0.gates import evaluate_gates
from mirage_persist.experiments.cv0.potency_screen import PotencyResult, run_potency_screen
from mirage_persist.experiments.cv0.variance_estimation import VarianceResult, run_variance_estimation
from mirage_persist.run.run_context import RunContext


@dataclass
class CV0Report:
    determinism: dict
    capability: dict
    potency: dict
    variance: dict
    gates: list[dict]
    overall: str


def _mirror_to_artifacts(config: CV0Config, run_path: Path) -> None:
    dest_dir = Path(config.paths.artifacts_dir) / config.experiment_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(run_path, dest_dir / run_path.name)
    except Exception:
        pass


def _capability_report_md(config: CV0Config, cap: CapabilityResult, gates: list[GateResult], overall: str) -> str:
    lines = [
        f"# CV-0 Capability Report - {config.experiment_id}",
        "",
        f"Overall verdict: **{overall}**",
        "",
        "## Eligibility (per confirmatory cell)",
        "",
        "| cell | eligible checkpoints |",
        "|---|---|",
    ]
    for cell, n in sorted(cap.per_cell_eligible.items()):
        lines.append(f"| {cell} | {n} |")
    lines += [
        "",
        f"Total eligible checkpoints: **{cap.n_eligible_total}** "
        f"(need >= {config.gates.eligible_per_cell_min} per cell)",
        "",
        "## Invalid-action rate (per model family)",
        "",
        f"| family | invalid-action rate | clears <= {config.gates.invalid_action_max:.2f} |",
        "|---|---|---|",
    ]
    for fam, rate in sorted(cap.per_family_invalid_rate.items()):
        ok = "yes" if rate <= config.gates.invalid_action_max else "no"
        lines.append(f"| {fam} | {rate:.3f} | {ok} |")
    lines += [
        "",
        f"Families clearing the invalid floor: {cap.families_clearing_invalid_floor}",
        "",
        "## Gate results",
        "",
        "| gate | status | detail |",
        "|---|---|---|",
    ]
    for g in gates:
        lines.append(f"| {g.name} | {g.status.value} | {g.detail} |")
    lines.append("")
    return "\n".join(lines)


def run_cv0(config: CV0Config, ctx: RunContext) -> CV0Report:
    adapter = build_adapter(config)
    backends = build_backends(config)
    scaffolds = build_scaffolds(config)

    for _, backend in backends:
        ctx.add_model_provenance(backend.provenance())

    run_dir = ctx.run_dir
    with EventWriter(run_dir / "events.jsonl") as events:
        # (a) determinism audit on greedy base checkpoints
        print("[cv0] collecting checkpoints (greedy base rollouts)...")
        records = collect_checkpoints(
            adapter, config, backends, scaffolds,
            greedy=True, max_checkpoints_total=config.determinism_audit.n_checkpoints, tag_prefix="det",
        )
        print(f"[cv0] determinism audit over {len(records)} checkpoints...")
        det: DeterminismResult = run_determinism_audit(adapter, config, records, event_writer=events)

        # (b) capability screen
        print("[cv0] capability screen...")
        cap: CapabilityResult = run_capability_screen(adapter, config, backends, scaffolds, event_writer=events)

        # (c) potency screen
        print("[cv0] potency screen...")
        pot: PotencyResult = run_potency_screen(adapter, config, backends, scaffolds, event_writer=events)

    # (d) variance components + re-solved power table
    print("[cv0] variance estimation + power table...")
    var: VarianceResult = run_variance_estimation(config, det.outcome_samples)

    gates = evaluate_gates(config, det, cap, pot)
    overall = overall_status(gates).value

    # ---- artifacts ------------------------------------------------------ #
    da = config.determinism_audit
    branch_key = f"Y_branch_{da.branch_horizon_for_epsilon}"

    envelope = {
        "experiment_id": config.experiment_id,
        "digest_equality_rate": det.digest_equality_rate,
        "n_checkpoints": det.n_checkpoints,
        "epsilon": {k: v.to_dict() for k, v in det.null_summaries.items()},
        "per_cell": det.per_cell,
    }
    p_env = ctx.write_json("envelope.json", envelope)
    _mirror_to_artifacts(config, p_env)

    fig = figure_s1(
        det.per_checkpoint_nulls, run_dir / "figure_s1_twin_split_null.png",
        branch_mean_threshold=config.gates.twin_null_branch_mean_max,
        branch_p95_threshold=config.gates.twin_null_branch_p95_max, branch_key=branch_key,
    )
    ctx.record_artifact(fig)
    _mirror_to_artifacts(config, fig)

    p_elig = write_parquet(cap.eligibility_rows, run_dir / "eligibility.parquet")
    ctx.record_artifact(p_elig)
    _mirror_to_artifacts(config, p_elig)

    md = _capability_report_md(config, cap, gates, overall)
    p_md = run_dir / "capability_report.md"
    p_md.write_text(md, encoding="utf-8")
    ctx.record_artifact(p_md)
    _mirror_to_artifacts(config, p_md)

    ctx.write_json("potency.json", pot.to_dict())
    ctx.write_json("variance_power.json", var.to_dict())

    verdict = {
        "experiment_id": config.experiment_id,
        "overall": overall,
        "gates": [g.to_dict() for g in gates],
        "config_hash": ctx.config_hash,
    }
    p_verdict = ctx.write_json("cv0_verdict.json", verdict)
    _mirror_to_artifacts(config, p_verdict)
    ctx.set_gate_verdict(verdict)

    # ---- console summary ------------------------------------------------ #
    print("\n===== CV-0 VERDICT =====")
    for g in gates:
        print(f"  [{g.status.value:4}] {g.name}: {g.detail}")
    print(f"  OVERALL: {overall}")
    print(f"  digest_equality={det.digest_equality_rate:.3f} eligible_total={cap.n_eligible_total} "
          f"sigma_tau={var.sigma_tau_used:.3f}")
    print("========================\n")

    return CV0Report(
        determinism=det.to_dict(),
        capability=cap.to_dict(),
        potency=pot.to_dict(),
        variance=var.to_dict(),
        gates=[g.to_dict() for g in gates],
        overall=overall,
    )


__all__ = ["run_cv0", "CV0Report"]
