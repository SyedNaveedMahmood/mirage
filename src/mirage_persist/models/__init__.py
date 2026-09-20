"""Backend-agnostic LLM elements.

A ``Backend`` produces AgentDojo ``BasePipelineElement`` LLMs on demand. Three are
provided: ``mock`` (scripted, deterministic, GPU-free), ``hf`` (transformers
in-process), and ``openai-compatible`` (any vLLM/llama.cpp/LM Studio/Ollama server).

Seed and sampling ride the ``extra_args`` carrier (ABI-safe, threaded through every
AgentDojo ``query``); the loop sets ``extra_args["mirage_seed"]``/``["mirage_greedy"]``
before each decision, and elements append token usage to ``extra_args["mirage_usage"]``.
"""

from __future__ import annotations

from mirage_persist.models.base import (
    EXTRA_GREEDY,
    EXTRA_SEED,
    EXTRA_USAGE,
    Backend,
    TaskContext,
    build_backend,
    drain_usage,
)

__all__ = [
    "Backend",
    "TaskContext",
    "build_backend",
    "drain_usage",
    "EXTRA_SEED",
    "EXTRA_GREEDY",
    "EXTRA_USAGE",
]
