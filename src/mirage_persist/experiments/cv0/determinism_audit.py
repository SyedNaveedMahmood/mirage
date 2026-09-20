"""CV-0(a): determinism audit -> the epsilon floor.

Per checkpoint: (1) two untreated greedy twin continuations -> trajectory-digest
equality; (2) two twin restores x K stochastic continuations -> the twin-split null of
every outcome (the epsilon floor). Aggregated over checkpoints this produces the
reproducibility envelope and Figure S1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.cv0.common import CPRecord, emit_continuation_event, seed_schedule
from mirage_persist.outcomes.schema import ACTION_CLASSES
from mirage_persist.outcomes.scoring import score_continuation
from mirage_persist.stats.twin_split import NullSummary, summarize_null, twin_split_null_js, twin_split_null_scalar
from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.continuation import continue_branch, continue_many


@dataclass
class DeterminismResult:
    n_checkpoints: int
    n_digest_equal: int
    digest_equality_rate: float
    null_summaries: dict[str, NullSummary]
    per_checkpoint_nulls: dict[str, list[float]]
    per_cell: dict[str, dict[str, float]]
    outcomes: list[str] = field(default_factory=list)
    # outcome -> {checkpoint_id -> [values across the 2K stochastic continuations]}
    outcome_samples: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_checkpoints": self.n_checkpoints,
            "n_digest_equal": self.n_digest_equal,
            "digest_equality_rate": round(self.digest_equality_rate, 6),
            "epsilon": {k: v.to_dict() for k, v in self.null_summaries.items()},
            "per_cell": self.per_cell,
            "outcomes": self.outcomes,
        }


def _action_class_labels(rows: list) -> list[str]:
    labels: list[str] = []
    for ov in rows:
        for cls, n in ov.action_class_counts.items():
            labels.extend([cls] * n)
    return labels


def run_determinism_audit(
    adapter: DojoAdapter,
    config: CV0Config,
    records: list[CPRecord],
    *,
    event_writer=None,
) -> DeterminismResult:
    sched = seed_schedule(config)
    da = config.determinism_audit
    horizons = config.horizons
    scalar_outcomes = [f"Y_branch_{h}" for h in horizons] + ["Y_prog_T"]

    n_equal = 0
    per_cp_nulls: dict[str, list[float]] = {name: [] for name in scalar_outcomes}
    per_cp_nulls["D_act_js"] = []
    per_cell: dict[str, dict[str, float]] = {}
    # per-checkpoint outcome samples for CV-0(d) variance components
    sample_outcomes = scalar_outcomes + ["Y_sec_T", "invalid_action_rate"]
    outcome_samples: dict[str, dict[str, list[float]]] = {name: {} for name in sample_outcomes}

    for rec in records:
        cp = rec.checkpoint
        cell_key = rec.cell.key
        cell_stats = per_cell.setdefault(cell_key, {"n": 0.0, "n_equal": 0.0})

        # (1) greedy twin digest equality
        twin_a = continue_branch(
            adapter, rec.backend, cp, branch_id="det_twinA", replicate=0,
            seed=sched.seed_for(cp.checkpoint_id, "det_twinA", 0),
            horizon=da.greedy_horizon, budget_config=config.budget, greedy=True,
        )
        twin_b = continue_branch(
            adapter, rec.backend, cp, branch_id="det_twinB", replicate=0,
            seed=sched.seed_for(cp.checkpoint_id, "det_twinB", 0),
            horizon=da.greedy_horizon, budget_config=config.budget, greedy=True,
        )
        equal = twin_a.trajectory_digest == twin_b.trajectory_digest
        n_equal += int(equal)
        cell_stats["n"] += 1
        cell_stats["n_equal"] += int(equal)

        # (2) twin-split null over K stochastic continuations per twin
        conts_a = continue_many(
            adapter, rec.backend, cp, branch_id="null_A", seed_schedule=sched,
            K=da.K, horizon=da.stochastic_horizon, budget_config=config.budget,
        )
        conts_b = continue_many(
            adapter, rec.backend, cp, branch_id="null_B", seed_schedule=sched,
            K=da.K, horizon=da.stochastic_horizon, budget_config=config.budget,
        )
        rows_a = [score_continuation(adapter, cp, c, horizons=horizons) for c in conts_a]
        rows_b = [score_continuation(adapter, cp, c, horizons=horizons) for c in conts_b]

        # rich per-continuation events (outcome + budget + digest) for every future
        for cont, ov in zip([*conts_a, *conts_b], [*rows_a, *rows_b], strict=True):
            emit_continuation_event(event_writer, cont, ov, role="score", intervention_status="absent")

        for h in horizons:
            va = [r.y_branch.get(h) for r in rows_a]
            vb = [r.y_branch.get(h) for r in rows_b]
            per_cp_nulls[f"Y_branch_{h}"].append(twin_split_null_scalar(va, vb))
        per_cp_nulls["Y_prog_T"].append(
            twin_split_null_scalar([r.y_prog_T for r in rows_a], [r.y_prog_T for r in rows_b])
        )
        per_cp_nulls["D_act_js"].append(
            twin_split_null_js(_action_class_labels(rows_a), _action_class_labels(rows_b), ACTION_CLASSES)
        )

        # per-checkpoint outcome samples (pool both twins) for variance components
        all_rows = rows_a + rows_b
        for name in sample_outcomes:
            if name.startswith("Y_branch_"):
                h = int(name.rsplit("_", 1)[1])
                vals = [r.y_branch.get(h) for r in all_rows]
            elif name == "Y_prog_T":
                vals = [r.y_prog_T for r in all_rows]
            elif name == "Y_sec_T":
                vals = [r.y_sec_T for r in all_rows]
            elif name == "invalid_action_rate":
                vals = [r.invalid_action_rate for r in all_rows]
            else:
                vals = []
            outcome_samples[name][cp.checkpoint_id] = [v for v in vals if v is not None]

        if event_writer is not None:
            event_writer.emit(
                checkpoint_id=cp.checkpoint_id,
                branch_id="det",
                step=cp.step,
                role="score",
                env_digest=cp.env_digest_value,
                extra={
                    "cell": cell_key,
                    "digest_equal": equal,
                    f"twin_null_Y_branch_{da.branch_horizon_for_epsilon}": per_cp_nulls[
                        f"Y_branch_{da.branch_horizon_for_epsilon}"
                    ][-1]
                    if f"Y_branch_{da.branch_horizon_for_epsilon}" in per_cp_nulls
                    else None,
                },
            )

    for stats in per_cell.values():
        stats["digest_equality_rate"] = stats["n_equal"] / stats["n"] if stats["n"] else float("nan")

    summaries = {name: summarize_null(name, vals) for name, vals in per_cp_nulls.items()}
    n = len(records)
    return DeterminismResult(
        n_checkpoints=n,
        n_digest_equal=n_equal,
        digest_equality_rate=(n_equal / n if n else float("nan")),
        null_summaries=summaries,
        per_checkpoint_nulls=per_cp_nulls,
        per_cell=per_cell,
        outcomes=scalar_outcomes + ["D_act_js"],
        outcome_samples=outcome_samples,
    )


__all__ = ["run_determinism_audit", "DeterminismResult"]
