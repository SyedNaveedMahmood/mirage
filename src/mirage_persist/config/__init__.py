"""Configuration schema, loader, and hashing for MIRAGE-Persist experiments.

Every experiment knob lives in a YAML config validated against the pydantic
models in :mod:`mirage_persist.config.schema`. The loader supports an ``extends``
key (base-config inheritance) and dotted-path CLI overrides, and every resolved
config is canonically hashed (:mod:`mirage_persist.config.hashing`) so the
``config_hash`` can be recorded in every run manifest.
"""

from __future__ import annotations

from mirage_persist.config.hashing import canonical_config_hash
from mirage_persist.config.loader import load_config, load_raw_yaml, resolve_config
from mirage_persist.config.schema import CV0Config, ExperimentConfig, ModelBackendConfig, ScaffoldConfig

__all__ = [
    "load_config",
    "load_raw_yaml",
    "resolve_config",
    "canonical_config_hash",
    "CV0Config",
    "ExperimentConfig",
    "ModelBackendConfig",
    "ScaffoldConfig",
]
