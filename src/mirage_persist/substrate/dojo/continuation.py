"""Rollout (checkpoint generation) and continuation (fork + K stochastic futures).

``rollout`` runs an untreated (or injected) trajectory and captures a checkpoint
before every decision. ``continue_branch`` forks one continuation from a checkpoint
with a given seed and horizon; ``continue_many`` runs the K continuations of a branch
under the published seed schedule (common random numbers across branches). Outcome
scoring is intentionally left to the caller (the ``outcomes`` layer) so this module
stays pure execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.types import ChatMessage

from mirage_persist.config.schema import BudgetConfig
from mirage_persist.models.base import Backend, TaskContext
from mirage_persist.run.seeds import SeedSchedule
from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.agent_loop import LoopResult, run_agent_loop
from mirage_persist.substrate.dojo.budget import BudgetMeter
from mirage_persist.substrate.dojo.checkpoint import Checkpoint, make_checkpoint_id
from mirage_persist.substrate.dojo.digest import trajectory_digest

if TYPE_CHECKING:
    from mirage_persist.scaffolds.base import Scaffold


class BranchLabel(str, Enum):
    """Six-arm taxonomy (design 4.3). CV-0 uses only N (never-treated) twins."""

    N = "N"  # never-treated reference
    TR = "TR"  # treated-then-removed (primary treated arm, CV-1)
    TK = "TK"  # treated-retained
    EC = "EC"  # equal-cost non-deceptive distraction
    BA = "BA"  # benign artifact (structure/salience control)
    NC = "NC"  # negative control (env mutated, no agent-visible evidence)


@dataclass
class RolloutResult:
    suite_name: str
    user_task_id: str
    rollout_tag: str
    checkpoints: list[Checkpoint]
    loop_result: LoopResult
    trajectory_digest: str
    injections: dict[str, str] = field(default_factory=dict)


@dataclass
class ContinuationResult:
    checkpoint_id: str
    branch_id: str
    replicate: int
    seed: int
    horizon: int
    greedy: bool
    messages: list[ChatMessage]
    env: TaskEnvironment
    pre_env: TaskEnvironment
    loop_result: LoopResult
    trajectory_digest: str
    n_prefix_messages: int = 0  # messages carried over from the checkpoint (post-actions follow)


def _make_task_ctx(adapter: DojoAdapter, suite_name: str, task_id: str, env: TaskEnvironment) -> TaskContext:
    task = adapter.user_task(suite_name, task_id)
    gt = adapter.ground_truth_calls(task, env)
    return TaskContext(suite_name=suite_name, task=task, ground_truth_calls=gt)


def rollout(
    adapter: DojoAdapter,
    backend: Backend,
    suite_name: str,
    user_task_id: str,
    *,
    budget_config: BudgetConfig,
    base_seed: int,
    rollout_tag: str,
    greedy: bool = False,
    injections: dict[str, str] | None = None,
    max_checkpoints: int | None = None,
    scaffold: Scaffold | None = None,
) -> RolloutResult:
    """Run a full trajectory, capturing a checkpoint before each decision."""
    user_task = adapter.user_task(suite_name, user_task_id)
    task_env, pre_env = adapter.init_env_for_task(suite_name, user_task, injections)
    runtime = adapter.runtime(suite_name)
    tools_executor = adapter.tools_executor()
    task_ctx = _make_task_ctx(adapter, suite_name, user_task_id, task_env)
    llm = backend.make_llm(task_ctx)
    system_message = scaffold.system_message(adapter) if scaffold is not None else None
    scaffold_state = scaffold.initial_scaffold_state() if scaffold is not None else {}
    messages = adapter.initial_messages(user_task, system_message=system_message)
    budget = BudgetMeter.fresh(budget_config)
    max_steps = scaffold.max_steps if scaffold is not None else budget_config.max_steps

    checkpoints: list[Checkpoint] = []

    def hook(step: int, msgs, env) -> None:
        if max_checkpoints is not None and len(checkpoints) >= max_checkpoints:
            return
        n_prior = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
        cid = make_checkpoint_id(suite_name, user_task_id, rollout_tag, step)
        checkpoints.append(
            Checkpoint.capture(
                checkpoint_id=cid,
                suite_name=suite_name,
                benchmark_version=adapter.benchmark_version,
                user_task_id=user_task_id,
                step=step,
                messages=list(msgs),
                env=env,
                pre_env=pre_env,
                n_prior_actions=n_prior,
                scaffold_state=scaffold_state,
                injections=injections or {},
                rollout_tag=rollout_tag,
                remaining_steps=max_steps - step,
            )
        )

    result = run_agent_loop(
        llm=llm,
        tools_executor=tools_executor,
        runtime=runtime,
        env=task_env,
        messages=messages,
        budget=budget,
        max_steps=max_steps,
        base_seed=base_seed,
        greedy=greedy,
        checkpoint_hook=hook,
    )
    return RolloutResult(
        suite_name=suite_name,
        user_task_id=user_task_id,
        rollout_tag=rollout_tag,
        checkpoints=checkpoints,
        loop_result=result,
        trajectory_digest=trajectory_digest(result.messages, result.env),
        injections=injections or {},
    )


def continue_branch(
    adapter: DojoAdapter,
    backend: Backend,
    checkpoint: Checkpoint,
    *,
    branch_id: str,
    replicate: int,
    seed: int,
    horizon: int,
    budget_config: BudgetConfig,
    greedy: bool = False,
) -> ContinuationResult:
    """Fork ONE continuation from ``checkpoint`` for ``horizon`` decisions."""
    env = checkpoint.restore_env()
    pre_env = checkpoint.restore_pre_env()
    messages = checkpoint.restore_messages()
    n_prefix = len(messages)
    runtime = adapter.runtime(checkpoint.suite_name)
    tools_executor = adapter.tools_executor()
    task_ctx = _make_task_ctx(adapter, checkpoint.suite_name, checkpoint.user_task_id, pre_env)
    llm = backend.make_llm(task_ctx)
    budget = BudgetMeter.fresh(budget_config)

    result = run_agent_loop(
        llm=llm,
        tools_executor=tools_executor,
        runtime=runtime,
        env=env,
        messages=messages,
        budget=budget,
        max_steps=horizon,
        base_seed=seed,
        greedy=greedy,
        start_step=checkpoint.step,
    )
    return ContinuationResult(
        checkpoint_id=checkpoint.checkpoint_id,
        branch_id=branch_id,
        replicate=replicate,
        seed=seed,
        horizon=horizon,
        greedy=greedy,
        messages=result.messages,
        env=result.env,
        pre_env=pre_env,
        loop_result=result,
        trajectory_digest=trajectory_digest(result.messages, result.env),
        n_prefix_messages=n_prefix,
    )


def continue_many(
    adapter: DojoAdapter,
    backend: Backend,
    checkpoint: Checkpoint,
    *,
    branch_id: str,
    seed_schedule: SeedSchedule,
    K: int,
    horizon: int,
    budget_config: BudgetConfig,
    greedy: bool = False,
    replicate_offset: int = 0,
) -> list[ContinuationResult]:
    """Run the K continuations of a branch under the published seed schedule (CRN)."""
    out: list[ContinuationResult] = []
    for k in range(replicate_offset, replicate_offset + K):
        seed = seed_schedule.seed_for(checkpoint.checkpoint_id, branch_id, k)
        out.append(
            continue_branch(
                adapter,
                backend,
                checkpoint,
                branch_id=branch_id,
                replicate=k,
                seed=seed,
                horizon=horizon,
                budget_config=budget_config,
                greedy=greedy,
            )
        )
    return out


__all__ = [
    "BranchLabel",
    "RolloutResult",
    "ContinuationResult",
    "rollout",
    "continue_branch",
    "continue_many",
]
