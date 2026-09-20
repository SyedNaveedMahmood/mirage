"""Config loading: YAML + ``extends`` inheritance + ref inlining + CLI overrides.

Resolution order (later wins):
  1. every file named by ``extends`` (a string or list of strings), depth-first;
  2. the config file itself;
  3. dotted-path overrides passed on the command line.

``models`` and ``scaffolds`` list entries may be strings referencing other YAML
files; they are loaded and inlined. Reference paths are resolved relative to the
referring file's directory first, then relative to the configs root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from mirage_persist.config.schema import CV0Config

TConfig = TypeVar("TConfig", bound=BaseModel)

# The configs/ directory at the repo root, used as a fallback resolution base.
_CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"

_REF_LIST_KEYS = ("models", "scaffolds")


def load_raw_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty file -> empty dict)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must be a mapping at top level, got {type(data)!r}")
    return data


def _resolve_ref(ref: str, referring_dir: Path) -> Path:
    """Resolve a string reference to a YAML file."""
    candidates = [
        referring_dir / ref,
        _CONFIGS_ROOT / ref,
        Path(ref),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Could not resolve config reference {ref!r} from {referring_dir} "
        f"(tried: {', '.join(str(c) for c in candidates)})"
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``. Dicts merge; scalars/lists replace."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _inline_refs(data: dict[str, Any], referring_dir: Path) -> dict[str, Any]:
    """Replace string entries in models/scaffolds lists with the referenced dicts."""
    out = dict(data)
    for key in _REF_LIST_KEYS:
        if key not in out or not isinstance(out[key], list):
            continue
        expanded: list[Any] = []
        for entry in out[key]:
            if isinstance(entry, str):
                ref_path = _resolve_ref(entry, referring_dir)
                expanded.append(load_raw_yaml(ref_path))
            else:
                expanded.append(entry)
        out[key] = expanded
    return out


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file resolving its ``extends`` chain (depth-first, later wins)."""
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"Circular 'extends' detected at {path}")
    seen = seen | {path}

    raw = load_raw_yaml(path)
    extends = raw.pop("extends", None)
    merged: dict[str, Any] = {}
    if extends is not None:
        refs = [extends] if isinstance(extends, str) else list(extends)
        for ref in refs:
            base_path = _resolve_ref(ref, path.parent)
            merged = _deep_merge(merged, _load_with_extends(base_path, seen))
    merged = _deep_merge(merged, raw)
    # Inline model/scaffold refs relative to *this* file's directory.
    merged = _inline_refs(merged, path.parent)
    return merged


def _coerce_scalar(text: str) -> Any:
    """Parse an override value using YAML scalar rules (int/float/bool/null/str/list)."""
    return yaml.safe_load(text)


def _apply_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data[a][b][c] = value`` for dotted_key 'a.b.c', creating dicts as needed."""
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def resolve_config(path: str | Path, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve a config file to a fully-merged plain dict (pre-validation)."""
    data = _load_with_extends(Path(path))
    for dotted_key, raw_value in (overrides or {}).items():
        _apply_override(data, dotted_key, _coerce_scalar(raw_value))
    return data


def load_config(
    path: str | Path,
    schema: type[TConfig] = CV0Config,  # type: ignore[assignment]
    overrides: dict[str, str] | None = None,
) -> TConfig:
    """Resolve and validate a config file into a pydantic model instance."""
    data = resolve_config(path, overrides)
    return schema.model_validate(data)


__all__ = ["load_raw_yaml", "resolve_config", "load_config"]
