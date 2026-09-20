"""Reset operator interface + the branch state it transforms."""

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.types import ChatMessage


class Carrier(str, Enum):
    C = "C"  # context (message history + KV cache)
    M = "M"  # memory (persistent notes / lesson store)
    E = "E"  # environment (files, emails, transactions, service state)
    B = "B"  # budget (tokens, tool calls, steps, retries, wall-clock)
    H = "H"  # scaffold history (plan text, task queue, retry counters)
    U = "U"  # tool state (caches, permissions, sessions)
    P = "P"  # process (loaded model, sampler state)
    L = "L"  # laundered content (intervention re-derived into agent-authored text)


@dataclass
class BranchState:
    """The mutable starting state of a continuation, on which resets operate."""

    env: TaskEnvironment
    messages: list[ChatMessage]
    scaffold_state: dict[str, Any] = field(default_factory=dict)
    budget_target: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> BranchState:
        return BranchState(
            env=self.env.model_copy(deep=True),
            messages=copy.deepcopy(self.messages),
            scaffold_state=copy.deepcopy(self.scaffold_state),
            budget_target=dict(self.budget_target),
        )


class ResetOperator(abc.ABC):
    """Restore one carrier of ``state`` to its value in the never-treated ``untreated`` state.

    Pairing the reset with the untreated reference is what enables the untreated-reset
    artifact control (design 4.5): the same operator applied to an N branch measures
    ``artifact(r)``.
    """

    carrier: Carrier
    implemented: bool = False

    @abc.abstractmethod
    def reset(self, state: BranchState, untreated: BranchState) -> BranchState: ...


class CV2Placeholder(ResetOperator):
    """A registered-but-not-yet-implemented carrier reset (lands in CV-2)."""

    def __init__(self, carrier: Carrier) -> None:
        self.carrier = carrier
        self.implemented = False

    def reset(self, state: BranchState, untreated: BranchState) -> BranchState:
        raise NotImplementedError(
            f"Carrier {self.carrier.value} reset is implemented in CV-2, not CV-0. "
            "It is registered so the carrier decomposition slots in without refactor."
        )


__all__ = ["Carrier", "BranchState", "ResetOperator", "CV2Placeholder"]
