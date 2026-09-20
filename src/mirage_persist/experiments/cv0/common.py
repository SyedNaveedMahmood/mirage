"""Shared CV-0 helpers: cell iteration, checkpoint generation, seed schedule."""

from __future__ import annotations

from dataclasses import dataclass

from mirage_persist.config.schema import CV0Config, ModelBackendConfig
from mirage_persist.experiments.base import Cell
from mirage_persist.models.base import Backend
from mirage_persist.run.seeds import SeedSchedule
from mirage_persist.scaffolds.base import Scaffold
from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.checkpoint import Checkpoint
from mirage_persist.substrate.dojo.continuation import ContinuationResult, RolloutResult, rollout
from mirage_persist.substrate.dojo.digest import env_digest


@dataclass
class CPRecord:
    """A checkpoint tagged with the (model, scaffold) cell that produced it."""

    cell: Cell
    model_cfg: ModelBackendConfig
    backend: Backend
    scaffold: Scaffold
    checkpoint: Checkpoint
    suite: str
    task_id: str


def seed_schedule(config: CV0Config) -> SeedSchedule:
    return SeedSchedule(config.seed.global_seed, config.seed.schedule_salt)


def emit_continuation_event(
    events,
    cont: ContinuationResult,
    ov,
    *,
    role: str = "score",
    intervention_status: str | None = None,
    intervention_version: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write one rich per-continuation event (no-op if ``events`` is None).

    Populates the schema's per-step fields (budget snapshot, env digest, seed, outcome
    summary) so the event stream is debuggable across every screen, not just the
    determinism audit.
    """
    if events is None:
        return
    budget = cont.loop_result.budget
    payload = {
        "Y_branch": {str(k): v for k, v in ov.y_branch.items()},
        "Y_prog_T": ov.y_prog_T,
        "Y_sec_T": ov.y_sec_T,
        "n_post_tool_calls": ov.n_post_tool_calls,
        "action_class_counts": ov.action_class_counts,
        "trajectory_digest": cont.trajectory_digest,
    }
    if extra:
        payload.update(extra)
    events.emit(
        checkpoint_id=cont.checkpoint_id,
        branch_id=cont.branch_id,
        replicate=cont.replicate,
        step=cont.loop_result.steps_run,
        role=role,
        seed=cont.seed,
        env_digest=env_digest(cont.env),
        budget_snapshot=budget.as_dict(),
        invalid=(ov.invalid_action_rate > 0),
        intervention_status=intervention_status,
        intervention_version=intervention_version,
        extra=payload,
    )


def iter_cells(
    config: CV0Config,
    backends: list[tuple[ModelBackendConfig, Backend]],
    scaffolds: list[Scaffold],
):
    """Yield (Cell, model_cfg, backend, scaffold) for the model x scaffold grid."""
    for model_cfg, backend in backends:
        for scaffold in scaffolds:
            cell = Cell(model_name=model_cfg.name, family=model_cfg.family or model_cfg.name, scaffold_name=scaffold.name)
            yield cell, model_cfg, backend, scaffold


def select_task_ids(adapter: DojoAdapter, suite: str, limit: int | None) -> list[str]:
    ids = adapter.user_task_ids(suite)
    return ids if limit is None else ids[:limit]


@dataclass
class _Stratum:
    cell: Cell
    model_cfg: ModelBackendConfig
    backend: Backend
    scaffold: Scaffold
    suite: str
    tasks: list[str]
    pos: int = 0


def collect_checkpoints(
    adapter: DojoAdapter,
    config: CV0Config,
    backends: list[tuple[ModelBackendConfig, Backend]],
    scaffolds: list[Scaffold],
    *,
    greedy: bool,
    max_checkpoints_total: int,
    max_per_rollout: int | None = None,
    tasks_per_suite: int | None = None,
    tag_prefix: str = "base",
    injections: dict[str, str] | None = None,
    on_rollout=None,
) -> list[CPRecord]:
    """Collect checkpoints, **round-robin balanced** across (cell x suite) strata.

    Earlier this iterated cell -> suite -> task and stopped at the cap, which
    front-loaded the first suite (e.g. all banking) and left the determinism envelope
    unrepresentative of later suites/scaffolds. We now take one task's rollout from each
    stratum per round, cycling until the cap is hit or all strata are exhausted, so the
    first N checkpoints are spread across every confirmatory cell and suite.

    ``greedy`` gives a canonical (reproducible) base trajectory for the determinism
    audit; the capability screen calls its own per-rollout loop with distinct seeds.
    """
    strata: list[_Stratum] = []
    for cell, model_cfg, backend, scaffold in iter_cells(config, backends, scaffolds):
        for suite in config.substrate.suites:
            strata.append(
                _Stratum(
                    cell=cell,
                    model_cfg=model_cfg,
                    backend=backend,
                    scaffold=scaffold,
                    suite=suite,
                    tasks=select_task_ids(adapter, suite, tasks_per_suite),
                )
            )

    cap_per_rollout = max_per_rollout if max_per_rollout is not None else config.capability_screen.max_checkpoints_per_rollout
    records: list[CPRecord] = []
    progressed = True
    while progressed and len(records) < max_checkpoints_total:
        progressed = False
        for st in strata:
            if len(records) >= max_checkpoints_total:
                break
            if st.pos >= len(st.tasks):
                continue
            progressed = True
            task_id = st.tasks[st.pos]
            st.pos += 1
            tag = f"{tag_prefix}/{st.cell.key}/{st.suite}/{task_id}"
            r: RolloutResult = rollout(
                adapter,
                st.backend,
                st.suite,
                task_id,
                budget_config=config.budget,
                base_seed=config.seed.global_seed,
                rollout_tag=tag,
                greedy=greedy,
                scaffold=st.scaffold,
                injections=injections,
                max_checkpoints=cap_per_rollout,
            )
            if on_rollout is not None:
                on_rollout(st.cell, st.model_cfg, st.backend, st.scaffold, st.suite, task_id, r)
            for cp in r.checkpoints:
                records.append(
                    CPRecord(
                        cell=st.cell,
                        model_cfg=st.model_cfg,
                        backend=st.backend,
                        scaffold=st.scaffold,
                        checkpoint=cp,
                        suite=st.suite,
                        task_id=task_id,
                    )
                )
                if len(records) >= max_checkpoints_total:
                    break
    return records


__all__ = [
    "CPRecord",
    "seed_schedule",
    "emit_continuation_event",
    "iter_cells",
    "select_task_ids",
    "collect_checkpoints",
]
