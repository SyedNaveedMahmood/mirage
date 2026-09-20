"""Checkpoint: an immutable, deep-copied pre-decision state.

A checkpoint is captured *immediately before* an agent decision (the message
history ends in a tool result or the initial user message), so a continuation
forked from it re-makes that decision fresh -- the pre-branch point the design
requires. ``restore_*`` returns independent deep copies so forks never alias.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.types import ChatMessage

from mirage_persist.substrate.dojo.digest import env_digest


def make_checkpoint_id(suite_name: str, task_id: str, rollout_tag: str, step: int) -> str:
    """Deterministic, restore-stable id (seeds depend on it, so it must not be random)."""
    return f"{suite_name}/{task_id}/{rollout_tag}/step{step:03d}"


@dataclass
class Checkpoint:
    checkpoint_id: str
    suite_name: str
    benchmark_version: str
    user_task_id: str
    step: int  # the decision index this checkpoint precedes
    messages: list[ChatMessage]  # deep-copied; ends in tool-result or user
    env: TaskEnvironment  # deep-copied live env at this point
    pre_env: TaskEnvironment  # deep-copied task-start baseline (for scoring)
    env_digest_value: str
    n_prior_actions: int  # tool-call turns already taken before this point
    scaffold_state: dict[str, Any] = field(default_factory=dict)  # S2 notes, etc.
    injections: dict[str, str] = field(default_factory=dict)  # {} => untreated trajectory
    rollout_tag: str = ""
    parent_checkpoint_id: str | None = None
    remaining_steps: int | None = None  # budget steps left when captured

    @classmethod
    def capture(
        cls,
        *,
        checkpoint_id: str,
        suite_name: str,
        benchmark_version: str,
        user_task_id: str,
        step: int,
        messages: list[ChatMessage],
        env: TaskEnvironment,
        pre_env: TaskEnvironment,
        n_prior_actions: int,
        scaffold_state: dict[str, Any] | None = None,
        injections: dict[str, str] | None = None,
        rollout_tag: str = "",
        parent_checkpoint_id: str | None = None,
        remaining_steps: int | None = None,
    ) -> Checkpoint:
        """Deep-copy the live state into an immutable checkpoint."""
        return cls(
            checkpoint_id=checkpoint_id,
            suite_name=suite_name,
            benchmark_version=benchmark_version,
            user_task_id=user_task_id,
            step=step,
            messages=copy.deepcopy(messages),
            env=env.model_copy(deep=True),
            pre_env=pre_env.model_copy(deep=True),
            env_digest_value=env_digest(env),
            n_prior_actions=n_prior_actions,
            scaffold_state=copy.deepcopy(scaffold_state or {}),
            injections=dict(injections or {}),
            rollout_tag=rollout_tag,
            parent_checkpoint_id=parent_checkpoint_id,
            remaining_steps=remaining_steps,
        )

    # -- independent restores (never alias across forks) ------------------ #
    def restore_env(self) -> TaskEnvironment:
        return self.env.model_copy(deep=True)

    def restore_pre_env(self) -> TaskEnvironment:
        return self.pre_env.model_copy(deep=True)

    def restore_messages(self) -> list[ChatMessage]:
        return copy.deepcopy(self.messages)

    def restore_scaffold(self) -> dict[str, Any]:
        return copy.deepcopy(self.scaffold_state)

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "suite": self.suite_name,
            "user_task_id": self.user_task_id,
            "step": self.step,
            "n_prior_actions": self.n_prior_actions,
            "env_digest": self.env_digest_value,
            "n_messages": len(self.messages),
            "remaining_steps": self.remaining_steps,
            "injected": bool(self.injections),
        }


__all__ = ["Checkpoint", "make_checkpoint_id"]
