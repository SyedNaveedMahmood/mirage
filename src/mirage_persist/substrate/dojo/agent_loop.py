"""The MIRAGE ReAct loop: decision -> tool execution, with a pre-decision hook.

We drive AgentDojo's LLM and ``ToolsExecutor`` elements directly rather than using
its ``ToolsExecutionLoop``, because we need the checkpoint boundary to sit
*before* each decision (history ending in an observation), which is the pre-branch
point the estimand requires. ``ToolsExecutionLoop`` instead starts from an assistant
turn, one decision too late. Reusing ``ToolsExecutor`` keeps tool execution, YAML
formatting, and invalid-call handling identical to AgentDojo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionsRuntime, TaskEnvironment
from agentdojo.types import ChatMessage

from mirage_persist.models.base import EXTRA_GREEDY, EXTRA_SEED, drain_usage
from mirage_persist.run.seeds import step_seed
from mirage_persist.substrate.dojo.budget import BudgetMeter, BudgetSnapshot

# A hook called immediately before decision `step`, with the current history and env.
CheckpointHook = Callable[[int, Sequence[ChatMessage], TaskEnvironment], None]


@dataclass
class StepEvent:
    step: int
    n_tool_calls: int
    invalid_actions: int
    prompt_tokens: int
    completion_tokens: int
    terminal: bool
    tool_names: list[str] = field(default_factory=list)
    had_error: list[bool] = field(default_factory=list)


@dataclass
class LoopResult:
    messages: list[ChatMessage]
    env: TaskEnvironment
    steps_run: int
    events: list[StepEvent]
    budget: BudgetSnapshot
    terminated: bool  # agent emitted a no-tool-call (final answer) decision
    exhausted_reason: str | None


def run_agent_loop(
    *,
    llm: BasePipelineElement,
    tools_executor: BasePipelineElement,
    runtime: FunctionsRuntime,
    env: TaskEnvironment,
    messages: list[ChatMessage],
    budget: BudgetMeter,
    max_steps: int,
    base_seed: int,
    greedy: bool = False,
    start_step: int = 0,
    checkpoint_hook: CheckpointHook | None = None,
    extra_args: dict[str, Any] | None = None,
) -> LoopResult:
    """Run up to ``max_steps`` decisions from the given history/env.

    Returns the extended messages, the mutated env, per-step events, and the
    final budget snapshot. Timeouts/budget-exhaustion end the loop as *outcomes*.
    """
    extra: dict[str, Any] = dict(extra_args or {})
    messages = list(messages)
    events: list[StepEvent] = []
    terminated = False
    step = start_step
    steps_run = 0

    while steps_run < max_steps:
        if budget.check_exhausted() is not None:
            break

        # Checkpoint boundary: about to make decision `step` (history ends in observation).
        if checkpoint_hook is not None:
            checkpoint_hook(step, messages, env)

        extra[EXTRA_SEED] = step_seed(base_seed, step)
        extra[EXTRA_GREEDY] = greedy

        _, runtime, env, messages, extra = llm.query("", runtime, env, messages, extra)  # type: ignore[assignment]
        budget.record_step()
        p_tok, c_tok = drain_usage(extra)
        budget.add_tokens(p_tok, c_tok)

        decision = messages[-1]
        tool_calls = decision.get("tool_calls")
        if not tool_calls:
            events.append(
                StepEvent(
                    step=step,
                    n_tool_calls=0,
                    invalid_actions=0,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    terminal=True,
                )
            )
            terminated = True
            break

        n_before = len(messages)
        _, runtime, env, messages, extra = tools_executor.query("", runtime, env, messages, extra)  # type: ignore[assignment]
        new_results = messages[n_before:]
        tool_names = [str(getattr(tc, "function", None)) for tc in tool_calls]
        had_error = [bool(m.get("error")) for m in new_results if m.get("role") == "tool"]
        invalid = sum(had_error)
        budget.record_tool_calls(len(tool_calls))
        budget.record_invalid_actions(invalid)
        events.append(
            StepEvent(
                step=step,
                n_tool_calls=len(tool_calls),
                invalid_actions=invalid,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                terminal=False,
                tool_names=tool_names,
                had_error=had_error,
            )
        )
        step += 1
        steps_run += 1

    return LoopResult(
        messages=messages,
        env=env,
        steps_run=steps_run,
        events=events,
        budget=budget.snapshot(),
        terminated=terminated,
        exhausted_reason=budget.exhausted_reason,
    )


__all__ = ["run_agent_loop", "StepEvent", "LoopResult", "CheckpointHook"]
