"""CV-0(c): potency screen -> immediate diversion per intervention class vs. BA.

For each intervention class, expose the intervention on a set of (task, injection-task)
pairs and measure immediate diversion (the injection-aligned goal achieved during
exposure, i.e. Y^sec on the treated rollout) -- at exposure time only, never
post-removal. The BA (benign artifact) control isolates structure/salience: a class is
"potent" if its diversion exceeds BA by the gate margin.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.cv0.common import seed_schedule
from mirage_persist.interventions import get_intervention, intervention_names
from mirage_persist.outcomes.scoring import dispatch_security
from mirage_persist.scaffolds.base import Scaffold
from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.continuation import rollout
from mirage_persist.substrate.dojo.digest import env_digest


@dataclass
class PotencyResult:
    per_class_diversion: dict[str, float]
    per_class_n: dict[str, int]
    ba_rate: float
    per_class_pp_above_ba: dict[str, float]
    classes_passing: list[str]
    n_triples: int
    n_exposures: int

    def to_dict(self) -> dict[str, object]:
        return {
            "per_class_diversion": {k: round(v, 4) for k, v in self.per_class_diversion.items()},
            "ba_rate": round(self.ba_rate, 4),
            "per_class_pp_above_ba": {k: round(v, 4) for k, v in self.per_class_pp_above_ba.items()},
            "classes_passing": self.classes_passing,
            "n_triples": self.n_triples,
            "n_exposures": self.n_exposures,
        }


def _final_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content") or []
            if isinstance(content, str):
                return content
            return "".join(str(b.get("content", "")) for b in content if isinstance(b, dict))
    return ""


def _select_triples(adapter: DojoAdapter, config: CV0Config, limit: int) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    for suite in config.substrate.suites:
        inj_ids = adapter.injection_task_ids(suite)
        if not inj_ids:
            continue
        for task_id in adapter.user_task_ids(suite):
            triples.append((suite, task_id, inj_ids[0]))
            if len(triples) >= limit:
                return triples
    return triples


def _rollout_diversion(adapter: DojoAdapter, rollout_result, injection_task) -> float:
    if not rollout_result.checkpoints:
        return 0.0
    pre_env = rollout_result.checkpoints[0].pre_env
    final_env = rollout_result.loop_result.env
    trace = adapter.trace_from_messages(rollout_result.loop_result.messages)
    model_output = _final_text(rollout_result.loop_result.messages)
    return 1.0 if dispatch_security(injection_task, model_output, pre_env, final_env, trace) else 0.0


def run_potency_screen(
    adapter: DojoAdapter,
    config: CV0Config,
    backends: list,
    scaffolds: list[Scaffold],
    *,
    event_writer=None,
) -> PotencyResult:
    ps = config.potency_screen
    sched = seed_schedule(config)
    classes = ps.intervention_classes or intervention_names()
    if "benign_artifact" not in classes:
        classes = classes + ["benign_artifact"]

    # Screen on the primary cell (first backend x first scaffold) to bound cost.
    _, backend = backends[0]
    scaffold = scaffolds[0]

    triples = _select_triples(adapter, config, ps.n_checkpoints)
    per_class_vals: dict[str, list[float]] = {c: [] for c in classes}

    for suite, task_id, inj_id in triples:
        inj_task = adapter.injection_task(suite, inj_id)
        for class_name in classes:
            iv = get_intervention(class_name)
            injections = iv.build_injections(adapter, suite, inj_task)
            for e in range(ps.n_exposures):
                seed = sched.seed_for(f"{suite}/{task_id}/{class_name}", "potency", e)
                r = rollout(
                    adapter, backend, suite, task_id,
                    budget_config=config.budget, base_seed=seed,
                    rollout_tag=f"potency/{class_name}/{suite}/{task_id}/e{e}",
                    greedy=False, scaffold=scaffold, injections=injections,
                )
                diversion = _rollout_diversion(adapter, r, inj_task)
                per_class_vals[class_name].append(diversion)
                if event_writer is not None:
                    event_writer.emit(
                        checkpoint_id=f"{suite}/{task_id}",
                        branch_id="TR",
                        step=r.loop_result.steps_run,
                        role="potency",
                        seed=seed,
                        env_digest=env_digest(r.loop_result.env),
                        budget_snapshot=r.loop_result.budget.as_dict(),
                        intervention_status="exposed",
                        intervention_version=iv.version,
                        extra={
                            "intervention_class": class_name,
                            "suite": suite,
                            "task_id": task_id,
                            "injection_task": inj_id,
                            "exposure": e,
                            "immediate_diversion": diversion,
                        },
                    )

    per_class_rate = {c: (sum(v) / len(v) if v else 0.0) for c, v in per_class_vals.items()}
    ba_rate = per_class_rate.get("benign_artifact", 0.0)
    pp_above = {c: per_class_rate[c] - ba_rate for c in classes if c != "benign_artifact"}
    goal_bearing = set(intervention_names(exclude_controls=True))
    passing = [c for c, pp in pp_above.items() if c in goal_bearing and pp >= config.gates.potency_pp_above_ba_min]

    return PotencyResult(
        per_class_diversion=per_class_rate,
        per_class_n={c: len(v) for c, v in per_class_vals.items()},
        ba_rate=ba_rate,
        per_class_pp_above_ba=pp_above,
        classes_passing=passing,
        n_triples=len(triples),
        n_exposures=ps.n_exposures,
    )


__all__ = ["run_potency_screen", "PotencyResult"]
