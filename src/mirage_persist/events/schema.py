"""The versioned event schema (design 3.1).

One event per (checkpoint, branch, step). Fields cover the observation/decision at
that step, its tool request, a hash of the tool result, the action class, subgoal
deltas, the env digest, a budget snapshot, and -- for treated branches -- the
intervention status, removal verification, reset spec, and seed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mirage_persist.version import EVENT_SCHEMA_VERSION


class Event(BaseModel):
    schema_version: str = EVENT_SCHEMA_VERSION
    ts: str  # ISO-8601 UTC
    checkpoint_id: str
    branch_id: str
    replicate: int | None = None
    step: int
    role: str  # "decision" | "observation" | "score" | "intervention" | "removal" | "reset"
    seed: int | None = None

    tool_request: list[dict[str, Any]] | None = None  # [{function, args}]
    tool_result_hash: str | None = None
    action_class: str | None = None
    invalid: bool | None = None

    subgoal_deltas: dict[str, bool] | None = None
    env_digest: str | None = None
    budget_snapshot: dict[str, Any] | None = None

    # treatment/reset bookkeeping (None for untreated CV-0 branches)
    intervention_status: str | None = None  # "exposed" | "removed" | "absent"
    intervention_version: str | None = None
    removal_verified: bool | None = None
    reset_spec: list[str] | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Event"]
