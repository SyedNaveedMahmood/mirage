"""Seeded OpenAI-compatible backend (vLLM / llama.cpp / LM Studio / Ollama).

Subclasses AgentDojo's ``OpenAILLM`` to (1) read seed/greedy/top_p from ``extra_args``
and pass them to the completion request (AgentDojo's version passes only temperature),
and (2) capture ``completion.usage`` into the budget meter. No AgentDojo edit required.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from openai._types import NOT_GIVEN
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from mirage_persist.config.schema import ModelBackendConfig, SamplingConfig
from mirage_persist.models.base import EXTRA_GREEDY, EXTRA_SEED, Backend, TaskContext, record_usage
from mirage_persist.run.provenance import ModelProvenance


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def _seeded_completion(
    client,
    model,
    messages,
    tools,
    *,
    temperature,
    top_p,
    seed,
    max_tokens,
    extra_body,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature,
        top_p=top_p if top_p is not None else NOT_GIVEN,
        seed=seed if seed is not None else NOT_GIVEN,
        max_tokens=max_tokens if max_tokens is not None else NOT_GIVEN,
        extra_body=extra_body or None,
    )


class SeededOpenAILLM(OpenAILLM):
    """OpenAILLM that honours MIRAGE's seed/greedy/top_p and logs token usage."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        sampling: SamplingConfig,
        reasoning_mode: str = "auto",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(client, model, temperature=sampling.temperature)
        self.sampling = sampling
        self.reasoning_mode = reasoning_mode
        # Resolved once by the backend from the model config (reasoning_mode +
        # chat_template_kwargs + request_extra_body); constant for the whole run.
        self.extra_body = dict(extra_body or {})

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        seed = extra_args.get(EXTRA_SEED)
        greedy = bool(extra_args.get(EXTRA_GREEDY, False))
        temperature = 0.0 if greedy else self.sampling.temperature
        top_p = 1.0 if greedy else self.sampling.top_p

        openai_messages = [_message_to_openai(m, self.model) for m in messages]
        openai_tools = [_function_to_openai(t) for t in runtime.functions.values()]
        completion = _seeded_completion(
            self.client, self.model, openai_messages, openai_tools,
            temperature=temperature, top_p=top_p, seed=seed, max_tokens=self.sampling.max_tokens,
            extra_body=dict(self.extra_body),
        )
        usage = getattr(completion, "usage", None)
        if usage is not None:
            record_usage(extra_args, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class OpenAICompatBackend(Backend):
    def __init__(self, config: ModelBackendConfig) -> None:
        super().__init__(config)
        base_url = config.base_url or (os.environ.get(config.base_url_env) if config.base_url_env else None)
        if not base_url:
            raise ValueError(
                f"openai-compatible backend needs a base_url (set {config.base_url_env!r} env var "
                "or config.base_url). Point it at your vLLM/llama.cpp/LM Studio/Ollama server."
            )
        api_key = (os.environ.get(config.api_key_env) if config.api_key_env else None) or "EMPTY"
        self._base_url = base_url
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = config.served_model_name or config.model_id

    def make_llm(self, task_ctx: TaskContext | None = None) -> BasePipelineElement:
        return SeededOpenAILLM(
            self.client,
            self.model_id,
            self.config.sampling,
            reasoning_mode=self.config.reasoning_mode,
            extra_body=self.config.resolved_extra_body(),
        )

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            slot=self.config.name,
            backend="openai-compatible",
            model_id=self.model_id,
            family=self.config.family,
            revision=self.config.revision,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            device=self.config.device,
            base_url=self._base_url,
            served_model_name=self.config.served_model_name,
            server_engine="vllm" if os.environ.get("VLLM_SERVER_VERSION") else None,
            server_version=os.environ.get("VLLM_SERVER_VERSION"),
            sampling=self.config.sampling.model_dump(),
            reasoning_mode=self.config.reasoning_mode,
            chat_template_kwargs=self.config.resolved_chat_template_kwargs(),
            request_extra_body=dict(self.config.request_extra_body),
            tool_call_parser=self.config.tool_call_parser,
        )


__all__ = ["SeededOpenAILLM", "OpenAICompatBackend"]
