"""GPU introspection and metering: device names, driver, peak VRAM, GPU-hours.

Designed to degrade gracefully: if torch/CUDA is unavailable (e.g. mock/CPU runs)
everything returns empty/zero and no exception propagates. Peak VRAM is read from
torch's allocator stats; GPU-hours is wall-clock-active time x active device count.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    index: int
    total_mib: float | None = None
    driver_version: str | None = None


def _nvidia_smi_query() -> list[GPUInfo]:
    """Best-effort GPU list via nvidia-smi (works even without torch)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[GPUInfo] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            total = float(parts[2])
        except ValueError:
            continue
        gpus.append(GPUInfo(name=parts[1], index=idx, total_mib=total, driver_version=parts[3]))
    return gpus


def list_gpus() -> list[GPUInfo]:
    """List visible GPUs, preferring torch (respects CUDA_VISIBLE_DEVICES) then nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            gpus: list[GPUInfo] = []
            driver = None
            smi = {g.index: g for g in _nvidia_smi_query()}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mib = props.total_memory / (1024**2)
                drv = smi.get(i).driver_version if smi.get(i) else driver
                gpus.append(GPUInfo(name=props.name, index=i, total_mib=total_mib, driver_version=drv))
            return gpus
    except Exception:
        pass
    return _nvidia_smi_query()


def cuda_runtime_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class GPUMeter:
    """Tracks peak VRAM per device and accumulates GPU-active seconds.

    Call :meth:`reset_peak` before a GPU workload and :meth:`mark_gpu_active`
    (or use it as a context manager) around the workload. GPU-hours is the
    accumulated active wall-clock time multiplied by the active device count.
    """

    gpus: list[GPUInfo] = field(default_factory=list_gpus)
    _active_seconds: float = 0.0
    _peak_mib: dict[int, float] = field(default_factory=dict)
    _t_enter: float | None = None

    def reset_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass

    def sample_peak(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mib = torch.cuda.max_memory_allocated(i) / (1024**2)
                    self._peak_mib[i] = max(self._peak_mib.get(i, 0.0), mib)
        except Exception:
            pass

    def __enter__(self) -> GPUMeter:
        self._t_enter = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._t_enter is not None:
            self._active_seconds += time.perf_counter() - self._t_enter
            self._t_enter = None
        self.sample_peak()

    @property
    def active_seconds(self) -> float:
        return self._active_seconds

    def gpu_hours(self) -> float:
        n = max(1, len(self.gpus)) if self.gpus else 0
        return (self._active_seconds / 3600.0) * n

    def peak_vram_mib(self) -> dict[str, float]:
        return {str(k): round(v, 1) for k, v in self._peak_mib.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "gpus": [g.__dict__ for g in self.gpus],
            "gpu_count": len(self.gpus),
            "cuda_available": cuda_available(),
            "cuda_runtime": cuda_runtime_version(),
            "gpu_active_seconds": round(self._active_seconds, 3),
            "gpu_hours": round(self.gpu_hours(), 5),
            "peak_vram_mib": self.peak_vram_mib(),
        }


__all__ = ["GPUInfo", "GPUMeter", "list_gpus", "cuda_available", "cuda_runtime_version"]
