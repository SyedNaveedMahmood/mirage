"""Write ``cluster_job_metadata.json`` for one Slurm CV-0 job.

Called from ``cv0_model.sbatch`` with the target path as its only argument; every
value arrives through ``MIRAGE_META_*`` / ``SLURM_*`` environment variables, so
no shell-quoted JSON is ever constructed by hand.

Credentials are never read here: the job passes the API key nowhere near this
script, and the environment keys below are an explicit allowlist rather than a
dump of ``os.environ``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Slurm facts worth keeping; all optional.
_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_QOS",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_SUBMIT_DIR",
    "SLURM_CLUSTER_NAME",
)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _read_json(path: str | None) -> object | None:
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _time_limit() -> str | None:
    """Requested walltime, read from inside the allocation (best effort)."""
    job_id = _env("SLURM_JOB_ID")
    if not job_id or shutil.which("scontrol") is None:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for field in out.stdout.split():
        if field.startswith("TimeLimit="):
            return field.split("=", 1)[1]
    return None


def build_metadata() -> dict[str, object]:
    return {
        "schema": "mirage-cluster-job-metadata/1",
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "job_mode": _env("MIRAGE_META_MODE"),
        "slurm": {key: _env(key) for key in _SLURM_KEYS},
        "requested_time_limit": _time_limit(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else _env("HOSTNAME"),
        "cuda_visible_devices": _env("CUDA_VISIBLE_DEVICES"),
        "model_profile": _env("MIRAGE_META_PROFILE"),
        "model_profile_description": _env("MIRAGE_META_PROFILE_DESC"),
        "model_id": _env("MIRAGE_META_MODEL_ID"),
        "model_revision": _env("MIRAGE_META_MODEL_REVISION"),
        "served_model_name": _env("MIRAGE_META_SERVED_MODEL"),
        "model_family": _env("MIRAGE_META_MODEL_FAMILY"),
        "tool_call_parser": _env("MIRAGE_META_TOOL_PARSER"),
        "reasoning_parser": _env("MIRAGE_META_REASONING_PARSER"),
        "reasoning_effort": _env("MIRAGE_META_REASONING_EFFORT"),
        "chat_template_path": _env("MIRAGE_META_CHAT_TEMPLATE"),
        "chat_template_sha256": _env("MIRAGE_META_CHAT_TEMPLATE_SHA256"),
        "vllm_version": _env("MIRAGE_META_VLLM_VERSION"),
        "vllm_port": _env("MIRAGE_META_PORT"),
        "vllm_launch_command_redacted": _env("MIRAGE_META_LAUNCH_CMD"),
        "vllm_startup_duration_s": _env("MIRAGE_META_STARTUP_S"),
        "cv0_config": _env("MIRAGE_META_CONFIG"),
        "v1_models": _read_json(_env("MIRAGE_META_MODELS_JSON")),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: write_job_metadata.py <output.json>", file=sys.stderr)
        return 2
    out_path = Path(argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(build_metadata(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
