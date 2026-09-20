"""Budget accounting: tokens, tool calls, steps, retries, wall-clock.

AgentDojo tracks none of these (it discards ``completion.usage`` and hard-codes a
15-step loop). MIRAGE meters them per branch so continuations can be budget-matched
(the estimand requires it) and so exhaustion is recorded as an *outcome*, never a
silent extension. Token counts are fed in by the LLM backends via ``add_tokens``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from mirage_persist.config.schema import BudgetConfig


@dataclass(frozen=True)
class BudgetSnapshot:
    steps: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    retries: int
    invalid_actions: int
    wall_clock_s: float
    exhausted_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "invalid_actions": self.invalid_actions,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "exhausted_reason": self.exhausted_reason,
        }


@dataclass
class BudgetMeter:
    """Mutable per-branch budget meter with enforceable caps."""

    config: BudgetConfig
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    retries: int = 0
    invalid_actions: int = 0
    _t0: float = field(default_factory=time.perf_counter)
    exhausted_reason: str | None = None

    # -- accounting ------------------------------------------------------- #
    def add_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += max(0, int(prompt))
        self.completion_tokens += max(0, int(completion))

    def record_step(self) -> None:
        self.steps += 1

    def record_tool_calls(self, n: int) -> None:
        self.tool_calls += max(0, int(n))

    def record_retry(self, n: int = 1) -> None:
        self.retries += max(0, int(n))

    def record_invalid_actions(self, n: int) -> None:
        self.invalid_actions += max(0, int(n))

    # -- limits ----------------------------------------------------------- #
    @property
    def wall_clock_s(self) -> float:
        return time.perf_counter() - self._t0

    def check_exhausted(self, *, extra_steps: int = 0) -> str | None:
        """Return the reason the budget is exhausted (and latch it), else None."""
        cfg = self.config
        if cfg.max_steps is not None and self.steps + extra_steps >= cfg.max_steps:
            self.exhausted_reason = self.exhausted_reason or "max_steps"
        elif cfg.max_tool_calls is not None and self.tool_calls >= cfg.max_tool_calls:
            self.exhausted_reason = self.exhausted_reason or "max_tool_calls"
        elif (
            cfg.max_completion_tokens is not None
            and self.completion_tokens >= cfg.max_completion_tokens
        ):
            self.exhausted_reason = self.exhausted_reason or "max_completion_tokens"
        elif cfg.wall_clock_s is not None and self.wall_clock_s >= cfg.wall_clock_s:
            self.exhausted_reason = self.exhausted_reason or "wall_clock"
        return self.exhausted_reason

    @property
    def exhausted(self) -> bool:
        return self.check_exhausted() is not None

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps=self.steps,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            tool_calls=self.tool_calls,
            retries=self.retries,
            invalid_actions=self.invalid_actions,
            wall_clock_s=self.wall_clock_s,
            exhausted_reason=self.exhausted_reason,
        )

    @classmethod
    def fresh(cls, config: BudgetConfig) -> BudgetMeter:
        return cls(config=config)


__all__ = ["BudgetMeter", "BudgetSnapshot"]
