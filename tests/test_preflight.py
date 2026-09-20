"""Functional preflight, driven entirely by mocked API responses.

No server is contacted and no model is loaded: `openai.OpenAI` is replaced by a
fake client that returns canned completions, which is enough to prove that the
preflight fails on a missing tool call / wrong served model and passes on a
well-formed one, and that the per-model ``extra_body`` really reaches the
request.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mirage_persist.config.loader import load_config
from mirage_persist.config.schema import BackendType, CV0Config, ModelBackendConfig

_ROOT = Path(__file__).resolve().parents[1]

# The preflight goes through AgentDojo's OpenAI message conversion, so the
# substrate must be installed for these tests to mean anything.
pytest.importorskip("agentdojo", reason="AgentDojo is required for the backend preflight path")

from openai.types.chat import ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion_message_tool_call import (  # noqa: E402
    ChatCompletionMessageToolCall,
    Function,
)

from mirage_persist.models import preflight as preflight_module  # noqa: E402


class FakeCompletions:
    """Records every request and replays scripted responses."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self._messages.pop(0) if len(self._messages) > 1 else self._messages[0]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )


class FakeClient:
    def __init__(self, messages, served=("Qwen/Qwen3.5-9B",)):
        self.completions = FakeCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=name) for name in served])
        )


def _text_message(text: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)


def _tool_message() -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="get_current_weather", arguments='{"city": "Paris"}'),
            )
        ],
    )


@pytest.fixture()
def patched_client(monkeypatch):
    """Install a fake OpenAI client and hand the test a handle on it."""
    holder: dict[str, FakeClient] = {}

    def _install(messages, served=("Qwen/Qwen3.5-9B",)):
        client = FakeClient(messages, served=served)
        holder["client"] = client
        monkeypatch.setattr(
            "mirage_persist.models.seeded_openai.openai.OpenAI",
            lambda **kwargs: client,
        )
        return client

    return _install


def _config(**kwargs) -> ModelBackendConfig:
    base = {
        "name": "M1",
        "backend": BackendType.OPENAI_COMPATIBLE,
        "model_id": "Qwen/Qwen3.5-9B",
        "served_model_name": "Qwen/Qwen3.5-9B",
        "base_url": "http://127.0.0.1:20001/v1",
    }
    return ModelBackendConfig(**{**base, **kwargs})


def test_preflight_passes_on_a_well_formed_server(patched_client):
    patched_client([_text_message("ready"), _tool_message()])
    report = preflight_module.run_preflight(_config(reasoning_mode="non-thinking"))
    assert report.ok, report.to_dict()
    assert [c["check"] for c in report.checks] == ["v1_models", "plain_completion", "tool_call"]
    assert "get_current_weather" in report.checks[-1]["detail"]


def test_preflight_fails_when_no_tool_call_is_parsed(patched_client):
    patched_client([_text_message("ready"), _text_message('{"name": "get_current_weather"}')])
    report = preflight_module.run_preflight(_config())
    assert not report.ok
    failed = [c for c in report.checks if not c["ok"]]
    assert failed[0]["check"] == "tool_call"
    assert "--tool-call-parser" in failed[0]["detail"]


def test_preflight_fails_on_a_served_model_mismatch(patched_client):
    patched_client([_text_message("ready"), _tool_message()], served=("some/other-model",))
    report = preflight_module.run_preflight(_config())
    assert not report.ok
    assert report.checks[0]["check"] == "v1_models"


def test_preflight_fails_when_the_server_rejects_the_request(patched_client):
    client = patched_client([_text_message("ready")])

    def _raise(**kwargs):
        raise ValueError("reasoning_effort is not a valid field")

    client.completions.create = _raise
    report = preflight_module.run_preflight(_config(chat_template_kwargs={"reasoning_effort": "low"}))
    assert not report.ok
    assert report.checks[-1]["check"] == "plain_completion"
    assert "reasoning_effort" in report.checks[-1]["detail"]


def test_gpt_oss_reasoning_effort_reaches_the_request_body(patched_client):
    client = patched_client([_text_message("ready"), _tool_message()], served=("openai/gpt-oss-20b",))
    config = _config(
        model_id="openai/gpt-oss-20b",
        served_model_name="openai/gpt-oss-20b",
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    report = preflight_module.run_preflight(config)
    assert report.ok, report.to_dict()
    for request in client.completions.requests:
        assert request["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_gpt_oss_preflight_has_room_to_finish_harmony_output(patched_client):
    client = patched_client(
        [_text_message("ready"), _tool_message()],
        served=("openai/gpt-oss-20b",),
    )
    config = _config(
        model_id="openai/gpt-oss-20b",
        served_model_name="openai/gpt-oss-20b",
        chat_template_kwargs={"reasoning_effort": "low"},
    )

    report = preflight_module.run_preflight(config)

    assert report.ok, report.to_dict()
    for request in client.completions.requests:
        assert request["max_tokens"] == preflight_module.PREFLIGHT_REASONING_MAX_TOKENS


def test_qwen_non_thinking_body_is_unchanged(patched_client):
    client = patched_client([_text_message("ready"), _tool_message()])
    preflight_module.run_preflight(_config(reasoning_mode="non-thinking"))
    for request in client.completions.requests:
        assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_preflight_sends_a_tool_schema_and_a_tiny_budget(patched_client):
    client = patched_client([_text_message("ready"), _tool_message()])
    preflight_module.run_preflight(_config())
    plain, tooled = client.completions.requests
    assert not plain["tools"]
    assert tooled["tools"], "the second probe must carry a JSON tool schema"
    assert tooled["tools"][0]["function"]["name"] == "get_current_weather"
    # A few dozen tokens, not the scientific max_tokens.
    assert plain["max_tokens"] == preflight_module.PREFLIGHT_MAX_TOKENS
    # Greedy + seeded so the probe is reproducible.
    assert plain["temperature"] == 0.0
    assert plain["seed"] == preflight_module.PREFLIGHT_SEED


def test_preflight_does_not_mutate_the_scientific_config(patched_client):
    patched_client([_text_message("ready"), _tool_message()])
    config = _config()
    original = config.model_dump()
    preflight_module.run_preflight(config)
    assert config.model_dump() == original
    assert config.sampling.max_tokens == 512


def test_preflight_skips_backends_without_a_server():
    config = ModelBackendConfig(name="mock", backend=BackendType.MOCK, model_id="mock-v1")
    report = preflight_module.run_preflight(config)
    assert report.ok
    assert report.checks[0]["skipped"] is True


def test_every_cluster_config_is_preflightable(patched_client, monkeypatch):
    """The four cluster configs all reach a PASS against a well-behaved server."""
    # The endpoint lives in the environment, exactly as the Slurm job sets it.
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:20001/v1")
    for name in (
        "cv0_qwen3p5_9b_100cp",
        "cv0_gemma4_26b_a4b_100cp",
        "cv0_gpt_oss_20b_100cp",
        "cv0_llama31_8b_100cp",
    ):
        cfg: CV0Config = load_config(_ROOT / "configs" / "cv0" / f"{name}.yaml", CV0Config)
        model = cfg.models[0]
        patched_client(
            [_text_message("ready"), _tool_message()],
            served=(model.served_model_name or model.model_id,),
        )
        report = preflight_module.run_preflight(model)
        assert report.ok, (name, report.to_dict())
