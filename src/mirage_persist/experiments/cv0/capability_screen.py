"""CV-0(b): capability screen -> eligible-checkpoint census.

For each (model, scaffold, suite, user task): run several rollouts, record reachability,
subgoal completion, and invalid-action rate; then apply the design-4.2 eligibility rule
(reachability, competence floor, remaining budget, >=2 viable continuations, exact twin
digest -- all pre-removal quantities). Produces eligibility rows and per-family invalid
rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.cv0.common import iter_cells, seed_schedule, select_task_ids
from mirage_persist.outcomes.subgoals import get_subgoals
from mirage_persist.scaffolds.base import Scaffold
from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.continuation import continue_branch, continue_many, rollout


@dataclass
class CapabilityResult:
    eligibility_rows: list[dict]
    per_cell_eligible: dict[str, int]
    per_family_invalid_rate: dict[str, float]
    per_family_families: dict[str, dict[str, float]]
    n_eligible_total: int
    families_clearing_invalid_floor: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_eligibility_rows": len(self.eligibility_rows),
            "n_eligible_total": self.n_eligible_total,
            "per_cell_eligible": self.per_cell_eligible,
            "per_family_invalid_rate": self.per_family_invalid_rate,
            "families_clearing_invalid_floor": self.families_clearing_invalid_floor,
        }


def _final_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content") or []
            if isinstance(content, str):
                return content
            return "".join(str(b.get("content", "")) for b in content if isinstance(b, dict))
    return ""


def _rollout_invalid_rate(r) -> float:
    b = r.loop_result.budget
    total = b.tool_calls
    return (b.invalid_actions / total) if total else 0.0


def run_capability_screen(
    adapter: DojoAdapter,
    config: CV0Config,
    backends: list,
    scaffolds: list[Scaffold],
    *,
    event_writer=None,
) -> CapabilityResult:
    cs = config.capability_screen
    rule = cs.eligibility
    sched = seed_schedule(config)

    rows: list[dict] = []
    per_cell_eligible: dict[str, int] = {}
    fam_invalid: dict[str, list[float]] = {}

    for cell, _model_cfg, backend, scaffold in iter_cells(config, backends, scaffolds):
        per_cell_eligible.setdefault(cell.key, 0)
        for suite in config.substrate.suites:
            for task_id in select_task_ids(adapter, suite, cs.max_tasks_per_suite):
                user_task = adapter.user_task(suite, task_id)
                sg = get_subgoals(suite, task_id, user_task)

                # rollouts_per_cell rollouts (distinct seeds via replicate in the tag)
                rollouts = []
                for rep in range(cs.rollouts_per_cell):
                    tag = f"cap/{cell.key}/{suite}/{task_id}/r{rep}"
                    base_seed = sched.seed_for(f"{suite}/{task_id}", cell.key, rep)
                    r = rollout(
                        adapter, backend, suite, task_id,
                        budget_config=config.budget, base_seed=base_seed, rollout_tag=tag,
                        greedy=False, scaffold=scaffold, max_checkpoints=cs.max_checkpoints_per_rollout,
                    )
                    rollouts.append(r)
                    fam_invalid.setdefault(cell.family, []).append(_rollout_invalid_rate(r))

                # reachability: count rollouts reaching each step
                reach: dict[int, int] = {}
                for r in rollouts:
                    for cp in r.checkpoints:
                        reach[cp.step] = reach.get(cp.step, 0) + 1

                # subgoal competence from the representative (first) rollout
                rep_rollout = rollouts[0]
                pre_env = rep_rollout.checkpoints[0].pre_env if rep_rollout.checkpoints else None
                n_subgoals = 0
                total_subgoals = len(sg.subgoals)
                if pre_env is not None:
                    flags = sg.satisfied(
                        _final_text(rep_rollout.loop_result.messages),
                        pre_env,
                        rep_rollout.loop_result.env,
                        adapter.trace_from_messages(rep_rollout.loop_result.messages),
                    )
                    n_subgoals = sum(flags)

                # eligibility per candidate checkpoint position (from the representative rollout)
                for cp in rep_rollout.checkpoints:
                    reach_count = reach.get(cp.step, 0)
                    remaining = cp.remaining_steps if cp.remaining_steps is not None else 0

                    # viable continuations: distinct trajectories among a small probe
                    probes = continue_many(
                        adapter, backend, cp, branch_id="probe", seed_schedule=sched,
                        K=cs.branch_probe_K, horizon=cs.branch_probe_horizon, budget_config=config.budget,
                    )
                    viable = len({p.trajectory_digest for p in probes})

                    # exact twin digest (greedy)
                    twin_exact = True
                    if rule.require_exact_twin_digest:
                        ta = continue_branch(adapter, backend, cp, branch_id="twinA", replicate=0,
                                             seed=sched.seed_for(cp.checkpoint_id, "twinA", 0),
                                             horizon=cs.branch_probe_horizon, budget_config=config.budget, greedy=True)
                        tb = continue_branch(adapter, backend, cp, branch_id="twinB", replicate=0,
                                             seed=sched.seed_for(cp.checkpoint_id, "twinB", 0),
                                             horizon=cs.branch_probe_horizon, budget_config=config.budget, greedy=True)
                        twin_exact = ta.trajectory_digest == tb.trajectory_digest

                    eligible = (
                        reach_count >= rule.min_reachable_rollouts
                        and n_subgoals >= rule.min_subgoals
                        and remaining >= rule.min_remaining_steps
                        and viable >= rule.min_viable_continuations
                        and (twin_exact or not rule.require_exact_twin_digest)
                    )
                    if eligible:
                        per_cell_eligible[cell.key] += 1

                    row = {
                        "checkpoint_id": cp.checkpoint_id,
                        "cell": cell.key,
                        "family": cell.family,
                        "model": cell.model_name,
                        "scaffold": cell.scaffold_name,
                        "suite": suite,
                        "task_id": task_id,
                        "step": cp.step,
                        "reach_count": reach_count,
                        "n_subgoals": n_subgoals,
                        "total_subgoals": total_subgoals,
                        "remaining_steps": remaining,
                        "viable_continuations": viable,
                        "twin_exact": twin_exact,
                        "eligible": eligible,
                    }
                    rows.append(row)
                    if event_writer is not None:
                        event_writer.emit(
                            checkpoint_id=cp.checkpoint_id,
                            branch_id="capability",
                            step=cp.step,
                            role="eligibility",
                            env_digest=cp.env_digest_value,
                            extra={k: row[k] for k in row if k != "checkpoint_id"},
                        )

    per_family_invalid = {f: (sum(v) / len(v) if v else float("nan")) for f, v in fam_invalid.items()}
    families_ok = [f for f, rate in per_family_invalid.items() if rate <= config.gates.invalid_action_max]
    n_eligible = sum(1 for row in rows if row["eligible"])

    return CapabilityResult(
        eligibility_rows=rows,
        per_cell_eligible=per_cell_eligible,
        per_family_invalid_rate=per_family_invalid,
        per_family_families={},
        n_eligible_total=n_eligible,
        families_clearing_invalid_floor=families_ok,
    )


__all__ = ["run_capability_screen", "CapabilityResult"]
