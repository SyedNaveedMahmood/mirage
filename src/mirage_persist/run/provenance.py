"""Run provenance: the full manifest recorded for every run.

Captures the field list the design requires (and then some): time/date, GPU name +
hours + VRAM, git commits (MIRAGE and the AgentDojo substrate), config hash, model,
tokenizer, substrate revision, seed, parent-child branch, terminal-log path, plus
hardware, dependency versions, determinism flags, and exit status.

Everything degrades gracefully -- a missing git binary or GPU never aborts a run.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mirage_persist.version import (
    EXPECTED_AGENTDOJO_SHA,
    EXPECTED_AGENTDOJO_VERSION,
    MANIFEST_SCHEMA_VERSION,
    __version__,
)

# Dependencies whose versions are load-bearing for reproducibility.
_TRACKED_DEPS = (
    "agentdojo",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "vllm",
    "openai",
    "numpy",
    "scipy",
    "statsmodels",
    "pandas",
    "pyarrow",
    "pydantic",
    "matplotlib",
    "mirage-persist",
)


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(repo_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_dir)
    sha = _run_git(repo, "rev-parse", "HEAD")
    branch = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git(repo, "status", "--porcelain")
    describe = _run_git(repo, "describe", "--tags", "--always", "--dirty")
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "describe": describe,
    }


def _dep_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_DEPS:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _agentdojo_repo_dir() -> Path | None:
    try:
        import agentdojo

        # src/agentdojo/__init__.py -> repo root is three parents up (src/agentdojo/.. -> src -> repo)
        return Path(agentdojo.__file__).resolve().parents[2]
    except Exception:
        return None


def pip_freeze_text() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class ModelProvenance(BaseModel):
    """One agent model's identity + tokenizer/sampling fingerprint (filled by backends)."""

    slot: str
    backend: str
    model_id: str
    family: str | None = None
    revision: str | None = None
    quantization: str | None = None
    dtype: str | None = None
    device: str | None = None
    weight_hash: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    served_model_name: str | None = None
    server_engine: str | None = None
    server_version: str | None = None
    reasoning_mode: str | None = None
    # Per-model request shaping actually sent to an OpenAI-compatible server, and
    # the server-side tool-call parser the job declared (advisory, from the config).
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    request_extra_body: dict[str, Any] = Field(default_factory=dict)
    tool_call_parser: str | None = None


class RunManifest(BaseModel):
    """The complete per-run provenance record, serialized to ``manifest.json``."""

    # -- identity / time
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    mirage_version: str = __version__
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    subcommand: str | None = None
    command_line: str = ""
    cwd: str = ""
    user: str | None = None
    started_at: str = ""
    ended_at: str | None = None
    duration_s: float | None = None
    exit_status: str | None = None  # "ok" | "error" | "killed"

    # -- hardware
    hostname: str = ""
    platform: str = ""
    os_version: str = ""
    python_version: str = ""
    python_impl: str = ""
    cpu: str = ""
    ram_total_gb: float | None = None
    gpu: dict[str, Any] = Field(default_factory=dict)  # from GPUMeter.as_dict()
    cuda_visible_devices: str | None = None
    determinism_flags: dict[str, Any] = Field(default_factory=dict)

    # -- code / deps
    mirage_git: dict[str, Any] = Field(default_factory=dict)
    substrate_git: dict[str, Any] = Field(default_factory=dict)
    substrate_expected_sha: str = EXPECTED_AGENTDOJO_SHA
    substrate_expected_version: str = EXPECTED_AGENTDOJO_VERSION
    substrate_matches_expected: bool | None = None
    dep_versions: dict[str, str | None] = Field(default_factory=dict)
    env_freeze_path: str | None = None
    env_freeze_sha256: str | None = None

    # -- config
    config_path: str | None = None
    config_hash: str | None = None
    resolved_config_path: str | None = None
    cli_overrides: dict[str, str] = Field(default_factory=dict)

    # -- substrate / experiment
    substrate_id: str | None = None
    agentdojo_benchmark_version: str | None = None
    suites: list[str] = Field(default_factory=list)
    scaffolds: list[str] = Field(default_factory=list)

    # -- models
    models: list[ModelProvenance] = Field(default_factory=list)

    # -- reproducibility
    global_seed: int | None = None
    seed_salt: str | None = None

    # -- outputs / diagnostics
    run_dir: str | None = None
    terminal_log_path: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    gate_verdict: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def collect_startup(experiment_id: str, subcommand: str | None) -> RunManifest:
    """Fill the fields knowable at run start (hardware, code, deps, determinism flags)."""
    uname = platform.uname()
    try:
        import psutil  # optional

        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_total_gb = None

    determinism_flags: dict[str, Any] = {}
    try:
        import torch

        determinism_flags = {
            "torch_deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        }
    except Exception:
        pass

    substrate_dir = _agentdojo_repo_dir()
    substrate_git = git_info(substrate_dir) if substrate_dir else {}
    matches = None
    if substrate_git.get("sha"):
        matches = substrate_git["sha"] == EXPECTED_AGENTDOJO_SHA

    return RunManifest(
        experiment_id=experiment_id,
        subcommand=subcommand,
        command_line=" ".join(sys.argv),
        cwd=os.getcwd(),
        user=_safe(getpass.getuser),
        started_at=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        os_version=uname.version,
        python_version=platform.python_version(),
        python_impl=platform.python_implementation(),
        cpu=uname.processor or platform.processor(),
        ram_total_gb=ram_total_gb,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        determinism_flags=determinism_flags,
        mirage_git=git_info(Path(__file__).resolve().parents[3]),
        substrate_git=substrate_git,
        substrate_matches_expected=matches,
        dep_versions=_dep_versions(),
    )


def write_env_freeze(path: str | Path) -> tuple[str, str]:
    """Write pip freeze to ``path`` and return (path, sha256)."""
    text = pip_freeze_text()
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(p), digest


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except Exception:
        return None


__all__ = ["RunManifest", "ModelProvenance", "collect_startup", "git_info", "write_env_freeze"]
