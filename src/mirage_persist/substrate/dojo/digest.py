"""Deterministic state and trajectory digests.

``env_digest`` canonically hashes an AgentDojo environment's *live* mutable state,
mirroring AgentDojo's own equality semantics: the redundant computed fields
(``sent``/``received``/``drafts``) and seed lists (``initial_*``) are dropped so the
digest tracks the derived live collections (``emails``/``events``/``files``) and the
rest of the state. Wall-clock timestamps set by tools (``timestamp`` on a sent email,
``last_modified`` on a cloud-drive file -- AgentDojo uses ``datetime.now()`` for these)
are also dropped: they are non-substantive environmental noise the decoder does not
control, so the design freezes them out of the reproducibility measurement. Twin
restores of the same checkpoint therefore hash identically up to such noise; only
decoder nondeterminism or a genuine state divergence can make continued twins differ.

``trajectory_digest`` hashes the ordered tool-call signature plus the final assistant
text plus the terminal env digest -- the primary signal for the CV-0(a) twin equality
check (it catches *any* divergence in actions, not just in the end state).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

# Field names dropped from env digests. The first block mirrors AgentDojo's own
# DeepDiff exclusions (computed/seed fields); the second block freezes wall-clock
# timestamps that AgentDojo tools stamp via datetime.now() (design: "timestamps frozen").
DEFAULT_VOLATILE_KEYS = frozenset(
    {
        "sent",
        "received",
        "drafts",
        "responses",
        "initial_emails",
        "initial_events",
        "initial_files",
        "model_fields_set",
        # wall-clock, tool-generated (non-substantive, decoder-independent):
        "timestamp",
        "last_modified",
    }
)


def _strip_volatile(obj: Any, volatile_keys: frozenset[str]) -> Any:
    """Recursively drop volatile keys from a JSON-able structure."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v, volatile_keys)
            for k, v in obj.items()
            if k not in volatile_keys
        }
    if isinstance(obj, list):
        return [_strip_volatile(v, volatile_keys) for v in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def env_digest(env: BaseModel, *, volatile_keys: frozenset[str] = DEFAULT_VOLATILE_KEYS) -> str:
    """Return the sha256 hex digest of an environment's canonical live state."""
    dumped = env.model_dump(mode="json")
    stripped = _strip_volatile(dumped, volatile_keys)
    return hashlib.sha256(_canonical_json(stripped).encode("utf-8")).hexdigest()


def canonicalize_args(args: Mapping[str, Any]) -> str:
    """Canonical JSON of a tool call's arguments (order-independent)."""
    return _canonical_json(dict(args))


def tool_call_signature(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Ordered list of (function_name, canonical_args) over all assistant tool calls."""
    sig: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = getattr(call, "function", None)
            raw_args = getattr(call, "args", None)
            if fn is None and isinstance(call, Mapping):
                fn = call.get("function")
                raw_args = call.get("args")
            sig.append((str(fn), canonicalize_args(raw_args or {})))
    return sig


def _final_assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            parts = []
            for block in content:
                text = block.get("content") if isinstance(block, Mapping) else getattr(block, "content", None)
                if text:
                    parts.append(str(text))
            return "".join(parts)
    return ""


def trajectory_digest(
    messages: Sequence[Mapping[str, Any]],
    env: BaseModel | None = None,
    *,
    include_final_text: bool = True,
) -> str:
    """Digest of (ordered tool-call signature, final assistant text, terminal env state)."""
    payload: dict[str, Any] = {"tool_calls": tool_call_signature(messages)}
    if include_final_text:
        payload["final_text"] = _final_assistant_text(messages)
    if env is not None:
        payload["env"] = env_digest(env)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "env_digest",
    "trajectory_digest",
    "canonicalize_args",
    "tool_call_signature",
    "DEFAULT_VOLATILE_KEYS",
]
