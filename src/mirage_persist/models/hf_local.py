"""In-process HuggingFace backend (tiny real models on the local GPU).

Loads a model+tokenizer once, formats messages+tools via the tokenizer chat template,
generates with a seedable RNG, and parses Qwen-style ``<tool_call>{...}</tool_call>``
blocks into AgentDojo ``FunctionCall``s. Token usage comes from the tokenizer. This is
the path that measures the REAL decoder-determinism envelope in CV-0(a).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from mirage_persist.config.schema import ModelBackendConfig
from mirage_persist.models.base import EXTRA_GREEDY, EXTRA_SEED, Backend, TaskContext, record_usage
from mirage_persist.run.provenance import ModelProvenance

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _to_hf_messages(messages: Sequence[ChatMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = get_text_content_as_str(m.get("content") or [])
        if role == "assistant":
            d: dict = {"role": "assistant", "content": text}
            tcs = m.get("tool_calls")
            if tcs:
                d["tool_calls"] = [
                    {"type": "function", "function": {"name": tc.function, "arguments": dict(tc.args)}}
                    for tc in tcs
                ]
            out.append(d)
        elif role == "tool":
            out.append({"role": "tool", "content": text, "name": getattr(m.get("tool_call"), "function", "")})
        else:
            out.append({"role": role, "content": text})
    return out


def _parse_tool_calls(text: str) -> tuple[str, list[FunctionCall]]:
    calls: list[FunctionCall] = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name and isinstance(args, dict):
            calls.append(FunctionCall(function=name, args=args, id=f"call_{i}"))
    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return cleaned, calls


class HFLocalLLM(BasePipelineElement):
    def __init__(self, backend: HFBackend) -> None:
        self._b = backend
        self.name = backend.config.name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        import torch

        b = self._b
        seed = extra_args.get(EXTRA_SEED)
        greedy = bool(extra_args.get(EXTRA_GREEDY, False))
        s = b.config.sampling

        hf_messages = _to_hf_messages(messages)
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        prompt = b.tokenizer.apply_chat_template(
            hf_messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = b.tokenizer(prompt, return_tensors="pt").to(b.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        gen_kwargs = dict(max_new_tokens=s.max_tokens, pad_token_id=b.tokenizer.eos_token_id)
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=s.temperature, top_p=s.top_p)

        with torch.inference_mode():
            out = b.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        completion_tokens = int(new_tokens.shape[0])
        text = b.tokenizer.decode(new_tokens, skip_special_tokens=True)

        record_usage(extra_args, prompt_tokens, completion_tokens)
        cleaned, calls = _parse_tool_calls(text)
        output = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(cleaned or "")],
            tool_calls=calls or None,
        )
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class HFBackend(Backend):
    def __init__(self, config: ModelBackendConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.dtype, torch.bfloat16)
        load_kwargs: dict = {"torch_dtype": dtype, "trust_remote_code": config.trust_remote_code}
        if config.revision:
            load_kwargs["revision"] = config.revision
        if config.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        else:
            load_kwargs["device_map"] = config.device if config.device != "cpu" else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
        if config.quantization != "4bit" and config.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self._resolved_revision = config.revision or getattr(self.model.config, "_commit_hash", None)

    def make_llm(self, task_ctx: TaskContext | None = None) -> BasePipelineElement:
        return HFLocalLLM(self)

    def provenance(self) -> ModelProvenance:
        chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        return ModelProvenance(
            slot=self.config.name,
            backend="hf",
            model_id=self.config.model_id,
            family=self.config.family,
            revision=self._resolved_revision,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            device=self.config.device,
            tokenizer_name=self.config.model_id,
            tokenizer_revision=self._resolved_revision,
            chat_template_hash=hashlib.sha256(chat_template.encode("utf-8")).hexdigest()[:16],
            sampling=self.config.sampling.model_dump(),
            reasoning_mode=self.config.reasoning_mode,
        )

    def close(self) -> None:
        try:
            import torch

            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["HFLocalLLM", "HFBackend"]
