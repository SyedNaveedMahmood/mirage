"""Mock backend: a scripted, deterministic, GPU-free agent.

This is the byte-exact engine gate. The default policy replays a task's AgentDojo
``ground_truth`` call sequence, so a mock rollout is a real, valid trajectory with
real branch structure and satisfied subgoals -- but with *zero* decoder
nondeterminism, which is exactly what the determinism audit needs to isolate the
engine's own reproducibility (it must be 100%).
"""

from __future__ import annotations

import abc
import copy
import json
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from mirage_persist.models.base import Backend, TaskContext, record_usage
from mirage_persist.run.provenance import ModelProvenance


def _count_prior_action_turns(messages: Sequence[ChatMessage]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))


def _estimate_tokens(text: str) -> int:
    # Deterministic, monotone-in-length proxy (mock accounting is not token science).
    return max(1, len(text) // 4)


def _prompt_token_estimate(messages: Sequence[ChatMessage]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            total += _estimate_tokens(get_text_content_as_str(content))
        elif isinstance(content, str):
            total += _estimate_tokens(content)
        for call in m.get("tool_calls") or []:
            total += _estimate_tokens(json.dumps(dict(getattr(call, "args", {})), default=str))
    return total


class MockPolicy(abc.ABC):
    @abc.abstractmethod
    def next_message(
        self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime, env: Env
    ) -> ChatAssistantMessage: ...


class GroundTruthPolicy(MockPolicy):
    """Emit the task's ground-truth calls in order, then a terminal answer."""

    def __init__(self, calls: Sequence[FunctionCall]) -> None:
        self.calls = list(calls)

    def next_message(
        self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime, env: Env
    ) -> ChatAssistantMessage:
        n = _count_prior_action_turns(messages)
        if n < len(self.calls):
            gt = self.calls[n]
            fc = FunctionCall(
                function=gt.function,
                args=copy.deepcopy(dict(gt.args)),
                id=f"call_{n}",  # deterministic id -> deterministic trajectory digest
            )
            return ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(f"Calling {gt.function}.")],
                tool_calls=[fc],
            )
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("Task complete.")],
            tool_calls=None,
        )


class TerminatePolicy(MockPolicy):
    """Immediately terminate with a fixed final answer (no actions)."""

    def next_message(
        self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime, env: Env
    ) -> ChatAssistantMessage:
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("Done.")],
            tool_calls=None,
        )


class MockLLM(BasePipelineElement):
    """AgentDojo pipeline element wrapping a deterministic :class:`MockPolicy`."""

    def __init__(self, policy: MockPolicy, name: str = "mock") -> None:
        self.policy = policy
        self.name = name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        output = self.policy.next_message(messages, runtime, env)
        prompt_tokens = _prompt_token_estimate(messages)
        completion_text = get_text_content_as_str(output.get("content") or [])
        completion_tokens = _estimate_tokens(completion_text) + 2 * len(output.get("tool_calls") or [])
        record_usage(extra_args, prompt_tokens, completion_tokens)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class MockBackend(Backend):
    def make_llm(self, task_ctx: TaskContext | None = None) -> BasePipelineElement:
        return MockLLM(self._policy_for(task_ctx), name=self.config.name)

    def _policy_for(self, task_ctx: TaskContext | None) -> MockPolicy:
        policy_name = self.config.mock_policy
        if policy_name == "ground_truth":
            calls = task_ctx.ground_truth_calls if task_ctx else []
            return GroundTruthPolicy(calls)
        if policy_name == "terminate":
            return TerminatePolicy()
        raise ValueError(f"Unknown mock_policy: {policy_name!r}")

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            slot=self.config.name,
            backend="mock",
            model_id=self.config.model_id,
            family=self.config.family,
            sampling=self.config.sampling.model_dump(),
            reasoning_mode=self.config.reasoning_mode,
        )


__all__ = ["MockLLM", "MockBackend", "MockPolicy", "GroundTruthPolicy", "TerminatePolicy"]
