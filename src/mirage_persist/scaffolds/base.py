"""Scaffold base + factory.

A scaffold provides the system message and the initial ``scaffold_state`` for a
rollout. Continuations inherit the system message via the checkpoint's message
history, so a scaffold only needs to act at rollout time.
"""

from __future__ import annotations

import abc
from typing import Any

from mirage_persist.config.schema import ScaffoldConfig, ScaffoldType
from mirage_persist.substrate.dojo.adapter import DojoAdapter


class Scaffold(abc.ABC):
    def __init__(self, config: ScaffoldConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def kind(self) -> ScaffoldType:
        return self.config.kind

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    @property
    def notes_enabled(self) -> bool:
        return self.config.notes_enabled

    @abc.abstractmethod
    def system_message(self, adapter: DojoAdapter) -> str: ...

    def initial_scaffold_state(self) -> dict[str, Any]:
        return {}


class ReActScaffold(Scaffold):
    """S1: thought/action/observation loop, no persistent notes (clean control)."""

    def system_message(self, adapter: DojoAdapter) -> str:
        return adapter.system_message()


class PlannerScaffold(Scaffold):
    """S2: plan-then-execute with a note store (makes M and H carriers real)."""

    _PREAMBLE = (
        "\n\nWork in a plan-then-execute style: first briefly outline a plan for the user's "
        "request, then carry it out step by step, keeping short notes on your progress and "
        "revising the plan as needed."
    )

    def system_message(self, adapter: DojoAdapter) -> str:
        return adapter.system_message() + self._PREAMBLE

    def initial_scaffold_state(self) -> dict[str, Any]:
        return {"plan": "", "notes": []}


def build_scaffold(config: ScaffoldConfig) -> Scaffold:
    if config.kind == ScaffoldType.S1_REACT:
        return ReActScaffold(config)
    if config.kind in (ScaffoldType.S2_PLANNER, ScaffoldType.S3_VERIFIER):
        # S3 (verifier) reuses the planner base for CV-0; its verifier module is CV-3.
        return PlannerScaffold(config)
    raise ValueError(f"Unknown scaffold kind: {config.kind!r}")


__all__ = ["Scaffold", "ReActScaffold", "PlannerScaffold", "build_scaffold"]
