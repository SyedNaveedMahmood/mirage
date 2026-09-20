"""Backend abstraction + the ``extra_args`` seed/usage protocol.

``extra_args`` conventions (all optional; absent => backend defaults):
    extra_args["mirage_seed"]   -> int    : the per-continuation seed
    extra_args["mirage_greedy"] -> bool   : force greedy decode (determinism audit)
    extra_args["mirage_usage"]  -> list   : elements APPEND {"prompt_tokens","completion_tokens"}
The agent loop sets seed/greedy before each decision and drains usage after.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall

from mirage_persist.config.schema import BackendType, ModelBackendConfig
from mirage_persist.run.provenance import ModelProvenance

EXTRA_SEED = "mirage_seed"
EXTRA_GREEDY = "mirage_greedy"
EXTRA_USAGE = "mirage_usage"


@dataclass
class TaskContext:
    """Per-task info a backend may need to build its LLM element (mock uses it)."""

    suite_name: str
    task: BaseUserTask | BaseInjectionTask
    ground_truth_calls: list[FunctionCall] = field(default_factory=list)


def drain_usage(extra_args: dict[str, Any]) -> tuple[int, int]:
    """Pop accumulated usage records from extra_args -> (prompt_tokens, completion_tokens)."""
    records = extra_args.pop(EXTRA_USAGE, [])
    p = sum(int(r.get("prompt_tokens", 0)) for r in records)
    c = sum(int(r.get("completion_tokens", 0)) for r in records)
    return p, c


def record_usage(extra_args: dict[str, Any], prompt_tokens: int, completion_tokens: int) -> None:
    extra_args.setdefault(EXTRA_USAGE, []).append(
        {"prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens)}
    )


class Backend(abc.ABC):
    """Produces LLM pipeline elements and reports its provenance."""

    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def family(self) -> str | None:
        return self.config.family

    @abc.abstractmethod
    def make_llm(self, task_ctx: TaskContext | None = None) -> BasePipelineElement:
        """Return an AgentDojo pipeline element that generates assistant turns."""

    @abc.abstractmethod
    def provenance(self) -> ModelProvenance:
        """Return the model/tokenizer/sampling fingerprint for the run manifest."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (GPU memory, HTTP clients). Default: no-op."""


def build_backend(config: ModelBackendConfig) -> Backend:
    """Instantiate the backend named by ``config.backend`` (heavy backends load lazily)."""
    if config.backend == BackendType.MOCK:
        from mirage_persist.models.mock import MockBackend

        return MockBackend(config)
    if config.backend == BackendType.HF:
        from mirage_persist.models.hf_local import HFBackend

        return HFBackend(config)
    if config.backend == BackendType.OPENAI_COMPATIBLE:
        from mirage_persist.models.seeded_openai import OpenAICompatBackend

        return OpenAICompatBackend(config)
    raise ValueError(f"Unknown backend type: {config.backend!r}")


__all__ = [
    "Backend",
    "TaskContext",
    "build_backend",
    "drain_usage",
    "record_usage",
    "EXTRA_SEED",
    "EXTRA_GREEDY",
    "EXTRA_USAGE",
]
