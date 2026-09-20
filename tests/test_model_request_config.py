"""Per-model request shaping and cross-model comparability of the CV-0 configs.

Two things are pinned here:

1. how ``reasoning_mode`` / ``chat_template_kwargs`` / ``request_extra_body``
   merge into one request body (including that Qwen's behaviour is unchanged);
2. that the four 100-checkpoint cluster configs differ **only** in model
   identity, experiment id and notes -- if that ever stops being true, the four
   runs stop being comparable.

No server and no GPU is involved: these are configuration-level tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mirage_persist.config.loader import load_config, resolve_config
from mirage_persist.config.schema import BackendType, CV0Config, ModelBackendConfig

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "configs"

CLUSTER_CONFIGS = {
    "qwen": _CONFIGS / "cv0" / "cv0_qwen3p5_9b_100cp.yaml",
    "gemma": _CONFIGS / "cv0" / "cv0_gemma4_26b_a4b_100cp.yaml",
    "gpt_oss": _CONFIGS / "cv0" / "cv0_gpt_oss_20b_100cp.yaml",
    "llama": _CONFIGS / "cv0" / "cv0_llama31_8b_100cp.yaml",
}

MODEL_CONFIGS = {
    "qwen": _CONFIGS / "models" / "openai_compat_vllm.yaml",
    "gemma": _CONFIGS / "models" / "openai_compat_vllm_gemma4_26b_a4b.yaml",
    "gpt_oss": _CONFIGS / "models" / "openai_compat_vllm_gpt_oss_20b.yaml",
    "llama": _CONFIGS / "models" / "openai_compat_vllm_llama31_8b.yaml",
}

# Fields a model arm is allowed to differ in. Everything else must match.
MODEL_SPECIFIC_KEYS = ("models", "experiment_id", "notes")


def _model(**kwargs) -> ModelBackendConfig:
    base = {"name": "M", "backend": BackendType.OPENAI_COMPATIBLE, "model_id": "x/y"}
    return ModelBackendConfig(**{**base, **kwargs})


# --------------------------------------------------------------------------- #
# merge precedence
# --------------------------------------------------------------------------- #
def test_reasoning_mode_auto_sends_nothing():
    model = _model()
    assert model.resolved_chat_template_kwargs() == {}
    assert model.resolved_extra_body() is None


def test_qwen_non_thinking_behaviour_is_unchanged():
    model = _model(reasoning_mode="non-thinking")
    assert model.resolved_chat_template_kwargs() == {"enable_thinking": False}
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_mode_behaviour_is_unchanged():
    model = _model(reasoning_mode="thinking")
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"enable_thinking": True}}


def test_chat_template_kwargs_are_forwarded():
    model = _model(chat_template_kwargs={"reasoning_effort": "low"})
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_chat_template_kwargs_merge_with_reasoning_mode():
    model = _model(reasoning_mode="non-thinking", chat_template_kwargs={"custom_flag": True})
    assert model.resolved_chat_template_kwargs() == {"enable_thinking": False, "custom_flag": True}


def test_request_extra_body_merges_alongside_template_kwargs():
    model = _model(reasoning_mode="non-thinking", request_extra_body={"top_k": 20})
    assert model.resolved_extra_body() == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_conflicting_reasoning_declarations_are_rejected_not_overwritten():
    with pytest.raises(ValidationError, match="enable_thinking"):
        _model(reasoning_mode="non-thinking", chat_template_kwargs={"enable_thinking": True})


def test_request_extra_body_cannot_shadow_backend_owned_keys():
    for key in ("chat_template_kwargs", "model", "messages", "tools", "tool_choice"):
        with pytest.raises(ValidationError):
            _model(request_extra_body={key: "x"})


def test_passthrough_values_are_not_coerced():
    model = _model(chat_template_kwargs={"flag": True, "count": 3, "ratio": 0.5, "name": "low"})
    kwargs = model.resolved_chat_template_kwargs()
    assert kwargs["flag"] is True
    assert isinstance(kwargs["count"], int) and not isinstance(kwargs["count"], bool)
    assert isinstance(kwargs["ratio"], float)
    assert kwargs["name"] == "low"


def test_resolved_extra_body_is_a_fresh_object():
    model = _model(reasoning_mode="non-thinking", request_extra_body={"top_k": 20})
    first = model.resolved_extra_body()
    assert first is not None
    first["top_k"] = 999
    assert model.resolved_extra_body() == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


# --------------------------------------------------------------------------- #
# the four model configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(MODEL_CONFIGS))
def test_model_config_is_pinned_and_valid(key):
    cfg = load_config(CLUSTER_CONFIGS[key], CV0Config)
    assert len(cfg.models) == 1
    model = cfg.models[0]
    assert model.backend == BackendType.OPENAI_COMPATIBLE
    assert model.revision and model.revision != "main"
    assert len(model.revision) == 40
    assert model.base_url is None  # endpoint comes from the environment
    assert model.base_url_env == "OPENAI_COMPATIBLE_BASE_URL"
    assert model.api_key_env == "OPENAI_COMPATIBLE_API_KEY"


def test_gemma_model_config():
    model = load_config(CLUSTER_CONFIGS["gemma"], CV0Config).models[0]
    assert model.model_id == "google/gemma-4-26B-A4B-it"
    assert model.revision == "462a98a12e28e2cbcfccaf78fe41e3e50235e6ae"
    assert model.family == "gemma4"
    assert model.dtype == "bfloat16"
    assert model.tool_call_parser == "gemma4"
    assert model.reasoning_mode == "non-thinking"
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_gpt_oss_model_config_sends_low_reasoning_effort():
    model = load_config(CLUSTER_CONFIGS["gpt_oss"], CV0Config).models[0]
    assert model.model_id == "openai/gpt-oss-20b"
    assert model.revision == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert model.family == "gpt-oss"
    assert model.tool_call_parser == "openai"
    # No enable_thinking boolean for this family: the effort string is what is sent.
    assert model.reasoning_mode == "auto"
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"reasoning_effort": "low"}}
    # Native MXFP4 must not be overridden with a bfloat16 weight cast.
    assert model.dtype == "auto"
    assert model.quantization == "mxfp4"


def test_llama_model_config():
    model = load_config(CLUSTER_CONFIGS["llama"], CV0Config).models[0]
    assert model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert model.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert model.family == "llama3.1"
    assert model.dtype == "bfloat16"
    assert model.tool_call_parser == "llama3_json"
    assert model.reasoning_mode == "auto"
    assert model.resolved_extra_body() is None  # no reasoning knob for this family


def test_qwen_model_config_is_unchanged():
    model = load_config(CLUSTER_CONFIGS["qwen"], CV0Config).models[0]
    assert model.model_id == "Qwen/Qwen3.5-9B"
    assert model.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert model.reasoning_mode == "non-thinking"
    assert model.chat_template_kwargs == {}
    assert model.request_extra_body == {}
    assert model.resolved_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_all_four_families_are_distinct():
    families = {load_config(path, CV0Config).models[0].family for path in CLUSTER_CONFIGS.values()}
    assert families == {"qwen3.5", "gemma4", "gpt-oss", "llama3.1"}


# --------------------------------------------------------------------------- #
# scientific comparability
# --------------------------------------------------------------------------- #
def test_the_four_cluster_configs_share_every_non_model_setting():
    resolved = {key: resolve_config(path) for key, path in CLUSTER_CONFIGS.items()}
    stripped = {}
    for key, data in resolved.items():
        stripped[key] = {k: v for k, v in data.items() if k not in MODEL_SPECIFIC_KEYS}

    reference = stripped["qwen"]
    for key, data in stripped.items():
        assert data == reference, f"{key} diverges from the Qwen reference run: {data} != {reference}"

    # And the settings that must be identical really are the ones from the run.
    assert reference["determinism_audit"]["n_checkpoints"] == 100
    assert reference["determinism_audit"]["K"] == 16
    assert reference["substrate"]["suites"] == ["banking", "slack", "travel", "workspace"]
    assert reference["gates"]["min_families_pass_capability"] == 1
    assert reference["gates"]["kill_min_families"] == 1


def test_experiment_ids_are_distinct():
    ids = {resolve_config(path)["experiment_id"] for path in CLUSTER_CONFIGS.values()}
    assert len(ids) == len(CLUSTER_CONFIGS)


@pytest.mark.parametrize("key", sorted(CLUSTER_CONFIGS))
def test_cluster_configs_validate_against_the_schema(key):
    cfg = load_config(CLUSTER_CONFIGS[key], CV0Config)
    assert cfg.determinism_audit.n_checkpoints == 100
    assert cfg.determinism_audit.K == 16
    assert {s.name for s in cfg.scaffolds} == {"S1", "S2"}
    assert cfg.capability_screen.rollouts_per_cell == 5
    assert cfg.potency_screen.n_checkpoints == 40
    assert cfg.seed.global_seed == 20260719
    assert cfg.horizons == [1, 3, 5, 10]
