"""Shared experiment scaffolding: build the population, iterate cells, gate status."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mirage_persist.config.schema import ExperimentConfig, ModelBackendConfig
from mirage_persist.models.base import Backend, build_backend
from mirage_persist.scaffolds.base import Scaffold, build_scaffold
from mirage_persist.substrate.dojo.adapter import DojoAdapter


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    KILL = "KILL"
    INFO = "INFO"


@dataclass
class GateResult:
    name: str
    status: GateStatus
    detail: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class Cell:
    """One (model family x scaffold) confirmatory cell; suite/task vary within."""

    model_name: str
    family: str
    scaffold_name: str

    @property
    def key(self) -> str:
        return f"{self.family}/{self.model_name}/{self.scaffold_name}"


def build_adapter(config: ExperimentConfig) -> DojoAdapter:
    return DojoAdapter(config.substrate.agentdojo_version)


def build_backends(config: ExperimentConfig) -> list[tuple[ModelBackendConfig, Backend]]:
    return [(m, build_backend(m)) for m in config.models]


def build_scaffolds(config: ExperimentConfig) -> list[Scaffold]:
    return [build_scaffold(s) for s in config.scaffolds]


def overall_status(gates: list[GateResult]) -> GateStatus:
    """KILL if any KILL; else FAIL if any FAIL; else PASS (INFO ignored)."""
    statuses = {g.status for g in gates}
    if GateStatus.KILL in statuses:
        return GateStatus.KILL
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    return GateStatus.PASS


__all__ = [
    "GateStatus",
    "GateResult",
    "Cell",
    "build_adapter",
    "build_backends",
    "build_scaffolds",
    "overall_status",
]
