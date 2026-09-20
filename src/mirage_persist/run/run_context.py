"""RunContext: the lifecycle wrapper around every experiment run.

Creates ``runs/<experiment_id>/<UTC-timestamp>_<config_hash8>/``, tees the terminal,
records provenance, seeds the RNGs, meters the GPU, and on exit writes
``manifest.json`` (+ ``resolved_config.yaml``, ``env_freeze.txt``). Experiments write
their event stream, outcome tables, figures, and reports into ``ctx.run_dir``.
"""

from __future__ import annotations

import json
import random
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import yaml
from pydantic import BaseModel

from mirage_persist.config.hashing import canonical_config_hash, short_config_hash
from mirage_persist.config.schema import BackendType, ExperimentConfig
from mirage_persist.run.gpu import GPUMeter
from mirage_persist.run.provenance import (
    ModelProvenance,
    RunManifest,
    collect_startup,
    write_env_freeze,
)
from mirage_persist.run.seeds import SeedSchedule
from mirage_persist.run.terminal_capture import TerminalCapture


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class RunContext:
    """Owns the run directory, provenance manifest, terminal capture, and metering."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        subcommand: str | None = None,
        config_path: str | Path | None = None,
        cli_overrides: dict[str, str] | None = None,
        deterministic_torch: bool = False,
    ) -> None:
        self.config = config
        self.config_path = str(config_path) if config_path else None
        self.cli_overrides = dict(cli_overrides or {})
        self.deterministic_torch = deterministic_torch
        self._subcommand = subcommand

        self.config_hash = canonical_config_hash(config)
        runs_root = Path(config.paths.runs_dir)
        self.run_dir = runs_root / config.experiment_id / f"{_timestamp()}_{short_config_hash(config)}"

        self.seed_schedule = SeedSchedule(config.seed.global_seed, config.seed.schedule_salt)
        self.gpu_meter = GPUMeter()
        self.manifest: RunManifest = RunManifest()  # replaced in __enter__
        self._terminal: TerminalCapture | None = None
        self._t_start: float = 0.0

    # -- lifecycle -------------------------------------------------------- #
    def __enter__(self) -> RunContext:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._terminal = TerminalCapture(self.run_dir / "terminal.log")
        self._terminal.__enter__()
        self._t_start = _perf()

        self._seed_everything()
        self.gpu_meter.reset_peak()

        self.manifest = collect_startup(self.config.experiment_id, subcommand=self._subcommand)
        self.manifest.run_dir = str(self.run_dir)
        self.manifest.terminal_log_path = str(self.run_dir / "terminal.log")
        self.manifest.config_path = self.config_path
        self.manifest.config_hash = self.config_hash
        self.manifest.cli_overrides = self.cli_overrides
        self.manifest.substrate_id = self.config.substrate.id
        self.manifest.agentdojo_benchmark_version = self.config.substrate.agentdojo_version
        self.manifest.suites = list(self.config.substrate.suites)
        self.manifest.scaffolds = [s.name for s in self.config.scaffolds]
        self.manifest.global_seed = self.config.seed.global_seed
        self.manifest.seed_salt = self.config.seed.schedule_salt

        # Persist the exact resolved config next to the manifest.
        resolved_path = self.run_dir / "resolved_config.yaml"
        resolved_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.manifest.resolved_config_path = str(resolved_path)

        env_path, env_sha = write_env_freeze(self.run_dir / "env_freeze.txt")
        self.manifest.env_freeze_path = env_path
        self.manifest.env_freeze_sha256 = env_sha

        print(f"[run] {self.config.experiment_id} :: {self.run_dir}")
        print(f"[run] config_hash={self.config_hash[:16]} run_id={self.manifest.run_id}")
        if self.manifest.substrate_matches_expected is False:
            msg = (
                f"substrate SHA {self.manifest.substrate_git.get('sha')} != expected "
                f"{self.manifest.substrate_expected_sha} (intentional upgrade? recorded, not fatal)"
            )
            print(f"[run][warn] {msg}")
            self.manifest.warnings.append(msg)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None:
            self.manifest.exit_status = "error"
            self.manifest.errors.append("".join(traceback.format_exception(exc_type, exc, tb)))
            print(f"[run][error] {exc_type.__name__ if exc_type else 'Error'}: {exc}")
        elif self.manifest.exit_status is None:
            self.manifest.exit_status = "ok"

        self.gpu_meter.sample_peak()
        self.manifest.ended_at = datetime.now(UTC).isoformat()
        self.manifest.duration_s = round(_perf() - self._t_start, 3)

        # GPU-hours. Two topologies:
        #  (1) in-process backend (hf): the client holds the GPU -> peak VRAM shows it.
        #  (2) server backend (openai-compatible / vLLM): the GPU work is in a SEPARATE
        #      process, so the client has no torch/VRAM. We then charge the run's
        #      wall-clock x visible-device count as a best-effort GPU-hours proxy (the
        #      job occupied the GPU for its duration). GPU name comes from nvidia-smi.
        # Mock/CPU runs match neither and record 0 GPU-hours.
        gpu_dict = self.gpu_meter.as_dict()
        peaks = gpu_dict.get("peak_vram_mib", {}) or {}
        in_process_gpu = bool(gpu_dict.get("cuda_available")) and any(v > 500 for v in peaks.values())
        server_gpu = self._uses_server_backend() and int(gpu_dict.get("gpu_count", 0)) > 0
        if (in_process_gpu or server_gpu) and self.manifest.duration_s is not None:
            n_dev = max(1, int(gpu_dict.get("gpu_count", 1)))
            gpu_dict["gpu_active_seconds"] = round(self.manifest.duration_s, 3)
            gpu_dict["gpu_hours"] = round((self.manifest.duration_s / 3600.0) * n_dev, 5)
            gpu_dict["gpu_hours_basis"] = "in_process_gpu_resident" if in_process_gpu else "server_side_wall_clock"
        self.manifest.gpu = gpu_dict

        self.write_manifest()
        print(f"[run] done status={self.manifest.exit_status} duration={self.manifest.duration_s}s")

        if self._terminal is not None:
            self._terminal.__exit__(exc_type, exc, tb)
        return False  # never suppress exceptions

    # -- helpers ---------------------------------------------------------- #
    def _uses_server_backend(self) -> bool:
        """True if any model runs on an out-of-process server (openai-compatible)."""
        return any(m.backend == BackendType.OPENAI_COMPATIBLE for m in self.config.models)

    def _seed_everything(self) -> None:
        seed = self.config.seed.global_seed
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.deterministic_torch:
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    def add_model_provenance(self, mp: ModelProvenance) -> None:
        self.manifest.models.append(mp)

    def record_artifact(self, path: str | Path) -> None:
        self.manifest.artifacts.append(str(path))

    def set_gate_verdict(self, verdict: dict | BaseModel) -> None:
        self.manifest.gate_verdict = verdict.model_dump() if isinstance(verdict, BaseModel) else verdict

    def write_manifest(self) -> Path:
        path = self.run_dir / "manifest.json"
        path.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_json(self, name: str, payload: dict | BaseModel) -> Path:
        path = self.run_dir / name
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        self.record_artifact(path)
        return path


def _perf() -> float:
    import time

    return time.perf_counter()


__all__ = ["RunContext"]
