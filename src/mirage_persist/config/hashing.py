"""Canonical hashing of resolved configs.

The ``config_hash`` is recorded in every run manifest so a run can be tied back
to the exact settings that produced it. The hash is deliberately stable across
machines: output-location fields (``paths``) and free-text ``notes`` are excluded
so that moving ``runs_dir`` or relabelling a run does not change the scientific
identity of the configuration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

# Top-level keys excluded from the scientific config hash.
_HASH_EXCLUDE_TOP_LEVEL = ("paths", "notes")


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def canonical_json(config: BaseModel | dict[str, Any], *, exclude_volatile: bool = True) -> str:
    """Return a canonical (sorted-key, compact) JSON string for hashing."""
    data = _to_plain(config)
    if not isinstance(data, dict):
        raise TypeError(f"canonical_json expects a mapping-like config, got {type(data)!r}")
    if exclude_volatile:
        data = {k: v for k, v in data.items() if k not in _HASH_EXCLUDE_TOP_LEVEL}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def canonical_config_hash(config: BaseModel | dict[str, Any], *, exclude_volatile: bool = True) -> str:
    """Full sha256 hex digest of the canonical config JSON."""
    payload = canonical_json(config, exclude_volatile=exclude_volatile).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_config_hash(config: BaseModel | dict[str, Any], *, length: int = 8) -> str:
    """First ``length`` hex chars of :func:`canonical_config_hash` (for run-dir names)."""
    return canonical_config_hash(config)[:length]


__all__ = ["canonical_json", "canonical_config_hash", "short_config_hash"]
