"""Functional API preflight for an OpenAI-compatible (vLLM) model server.

CV-0 is a multi-hour campaign; a wrong tool-call parser, a missing chat template
or a rejected reasoning setting must fail in seconds, not after eight hours. This
module sends three tiny requests to an already-running server:

    1. an ordinary chat completion (the request shaping is accepted at all);
    2. a chat completion carrying one simple JSON tool schema;
    3. the same response parsed back through *the same backend element MIRAGE
       uses* (``SeededOpenAILLM``), so a parser/template mismatch surfaces here.

It never touches the scientific configuration: the model config is copied with a
tiny ``max_tokens`` for the probes and the original object is left untouched.
No model-specific branching happens here — the per-model differences travel in
``chat_template_kwargs`` / ``request_extra_body`` on the model config, so
gpt-oss's ``reasoning_effort: low`` is exercised automatically by step 1 and 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mirage_persist.config.schema import BackendType, ModelBackendConfig

# Non-reasoning probes can be extremely small. Reasoning models such as
# gpt-oss need enough output budget to close their structured reasoning /
# Harmony envelope before a final answer or tool call can be parsed.
PREFLIGHT_MAX_TOKENS = 32
PREFLIGHT_REASONING_MAX_TOKENS = 512
PREFLIGHT_SEED = 1234

_PLAIN_PROMPT = "Reply with the single word: ready"
_TOOL_PROMPT = (
    "Call the get_current_weather tool for the city 'Paris'. "
    "Respond with the tool call only."
)


@dataclass
class PreflightReport:
    """Outcome of the preflight; serialised next to the job artifacts."""

    model_id: str
    served_model_name: str | None
    checks: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True

    def add(self, name: str, ok: bool, detail: str = "", **extra: Any) -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail, **extra})
        if not ok:
            self.ok = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "served_model_name": self.served_model_name,
            "ok": self.ok,
            "checks": self.checks,
        }


def get_current_weather(city: str) -> str:
    """Return a one-word weather report for a city.

    :param city: The city to report on.
    """
    return "sunny"


def _probe_config(config: ModelBackendConfig) -> ModelBackendConfig:
    """Return a probe-only copy without mutating the scientific configuration.

    Ordinary models use a deliberately tiny budget. Models that expose a
    reasoning_effort chat-template knob need a larger budget so their structured
    reasoning envelope can reach a terminal state before parsing.
    """
    max_tokens = (
        PREFLIGHT_REASONING_MAX_TOKENS
        if "reasoning_effort" in config.chat_template_kwargs
        else PREFLIGHT_MAX_TOKENS
    )
    sampling = config.sampling.model_copy(update={"max_tokens": max_tokens})
    return config.model_copy(update={"sampling": sampling})


def _user_message(text: str) -> Any:
    """A v0.1.35 user message (content is a list of blocks, not a bare string)."""
    from agentdojo.types import ChatUserMessage, text_content_block_from_string

    return ChatUserMessage(role="user", content=[text_content_block_from_string(text)])


def _extract_tool_calls(message: Any) -> list[Any]:
    """Tool calls from an AgentDojo assistant message (dict-like or attribute-like)."""
    if isinstance(message, dict):
        return list(message.get("tool_calls") or [])
    return list(getattr(message, "tool_calls", None) or [])


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # v0.1.35 stores content as a list of blocks.
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            parts.append(str(block.get("content", "")))
        else:
            parts.append(str(getattr(block, "content", "")))
    return "".join(parts)


def run_preflight(config: ModelBackendConfig) -> PreflightReport:
    """Run the three probes against the server described by ``config``."""
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime

    from mirage_persist.models.base import EXTRA_GREEDY, EXTRA_SEED, build_backend

    report = PreflightReport(model_id=config.model_id, served_model_name=config.served_model_name)

    if config.backend != BackendType.OPENAI_COMPATIBLE:
        report.add(
            "backend",
            True,
            f"backend {config.backend.value} needs no server preflight; skipped",
            skipped=True,
        )
        return report

    backend = build_backend(_probe_config(config))
    expected = config.served_model_name or config.model_id

    # -- 1. the server serves the model we think it serves ------------------
    try:
        served = [m.id for m in backend.client.models.list().data]  # type: ignore[attr-defined]
        ok = expected in served
        report.add("v1_models", ok, f"served={served}, expected={expected}")
    except Exception as exc:  # pragma: no cover - network/runtime failure path
        report.add("v1_models", False, f"{type(exc).__name__}: {exc}")
        return report

    # -- 2. an ordinary chat completion (validates request shaping) ---------
    # Sent through the same element MIRAGE uses, so extra_body (chat-template
    # kwargs such as gpt-oss reasoning_effort) is exercised exactly as in the run.
    llm = backend.make_llm()
    extra_args = {EXTRA_SEED: PREFLIGHT_SEED, EXTRA_GREEDY: True}
    empty_runtime = FunctionsRuntime([])
    try:
        _, _, _, messages, _ = llm.query(
            _PLAIN_PROMPT, empty_runtime, EmptyEnv(), [_user_message(_PLAIN_PROMPT)], extra_args
        )
        report.add(
            "plain_completion",
            True,
            f"response={_message_text(messages[-1])[:200]!r}",
            extra_body=config.resolved_extra_body(),
        )
    except Exception as exc:
        report.add("plain_completion", False, f"{type(exc).__name__}: {exc}")
        return report

    # -- 3. one JSON tool schema, parsed back by the same backend -----------
    tool_runtime = FunctionsRuntime([])
    tool_runtime.register_function(get_current_weather)
    extra_args = {EXTRA_SEED: PREFLIGHT_SEED, EXTRA_GREEDY: True}
    try:
        _, _, _, messages, _ = llm.query(
            _TOOL_PROMPT, tool_runtime, EmptyEnv(), [_user_message(_TOOL_PROMPT)], extra_args
        )
    except Exception as exc:
        report.add("tool_call", False, f"{type(exc).__name__}: {exc}")
        return report

    tool_calls = _extract_tool_calls(messages[-1])
    if not tool_calls:
        report.add(
            "tool_call",
            False,
            "the model returned no parsed tool call for an explicit tool request; "
            "check --tool-call-parser, --enable-auto-tool-choice and the chat template. "
            f"content={_message_text(messages[-1])[:400]!r}",
        )
        return report

    call = tool_calls[0]
    name = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
    report.add(
        "tool_call",
        True,
        f"parsed tool call {name!r} with arguments {json.dumps(args, default=str)[:200]}",
    )
    return report


__all__ = [
    "PreflightReport",
    "run_preflight",
    "get_current_weather",
    "PREFLIGHT_MAX_TOKENS",
    "PREFLIGHT_REASONING_MAX_TOKENS",
]
