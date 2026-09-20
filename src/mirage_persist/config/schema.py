"""Pydantic schema for MIRAGE-Persist experiment configurations.

Every knob of every experiment is declared here. Configs are ``extra="forbid"``
so a typo in a YAML key is a hard error rather than a silently-ignored setting —
essential when configs are the mutation surface for experiments.

The hierarchy:

    ExperimentConfig            (common: seed, substrate, budget, horizons, ...)
      └── CV0Config             (adds the four CV-0 sub-experiment blocks + gates)

Model and scaffold populations are lists so a single config sweeps M1..M4 x S1,S2.
In YAML, a ``models``/``scaffolds`` entry may be either an inline mapping or a
string path to another YAML file (resolved by the loader) — see loader.py.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

# Values allowed inside ``chat_template_kwargs`` / ``request_extra_body``. Strict
# scalars only: these end up verbatim in an OpenAI-compatible request body, so a
# silent bool->int (or int->str) coercion would change server behaviour. Anything
# richer belongs in a first-class schema field, not in a free-form passthrough.
JsonScalar = StrictBool | StrictInt | StrictFloat | StrictStr

# Request-body keys the backend itself owns; a model config may not shadow them.
_RESERVED_EXTRA_BODY_KEYS = frozenset({"chat_template_kwargs", "messages", "model", "tools", "tool_choice"})


class _Strict(BaseModel):
    """Base model that forbids unknown keys (catches config typos)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Models / sampling
# --------------------------------------------------------------------------- #
class BackendType(str, Enum):
    MOCK = "mock"
    HF = "hf"
    OPENAI_COMPATIBLE = "openai-compatible"


class SamplingConfig(_Strict):
    """Decoder controls. The determinism audit overrides these to greedy."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


class ModelBackendConfig(_Strict):
    """One agent model slot.

    ``family`` labels the pretraining lineage; the confirmatory unit is the
    *model family*, so screening groups by this rather than by ``model_id``.
    """

    name: str  # logical slot, e.g. "M1", "mock", "qwen2p5-1p5b"
    backend: BackendType
    model_id: str
    family: str | None = None
    revision: str | None = None
    quantization: str | None = None  # e.g. "4bit", "8bit", None
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    reasoning_mode: Literal["auto", "thinking", "non-thinking"] = "auto"

    # openai-compatible only (base_url may be given directly or via env var name)
    base_url: str | None = None
    base_url_env: str | None = "OPENAI_COMPATIBLE_BASE_URL"
    api_key_env: str | None = "OPENAI_COMPATIBLE_API_KEY"
    served_model_name: str | None = None

    # Generic per-model request shaping (openai-compatible backend).
    #
    # ``reasoning_mode`` only covers the Qwen-style ``enable_thinking`` boolean.
    # Families whose reasoning knob is not a boolean (gpt-oss uses the string
    # ``reasoning_effort``) declare it here instead, so no evaluator code needs a
    # per-model branch. ``chat_template_kwargs`` is forwarded to the server as
    # ``extra_body["chat_template_kwargs"]`` (vLLM applies it when rendering the
    # chat template); ``request_extra_body`` is merged into ``extra_body`` itself.
    chat_template_kwargs: dict[str, JsonScalar] = Field(default_factory=dict)
    request_extra_body: dict[str, JsonScalar] = Field(default_factory=dict)

    # Advisory only (recorded in provenance): the vLLM tool-call parser the server
    # was launched with. MIRAGE never sends it, but the manifest must show it.
    tool_call_parser: str | None = None

    # mock only: name of a registered deterministic policy
    mock_policy: str = "ground_truth"

    @model_validator(mode="after")
    def _default_family(self) -> ModelBackendConfig:
        if self.family is None:
            object.__setattr__(self, "family", self.name)
        return self

    @model_validator(mode="after")
    def _no_silent_overwrites(self) -> ModelBackendConfig:
        """Reject configs where two knobs would fight over the same request field."""
        if self.reasoning_mode != "auto" and "enable_thinking" in self.chat_template_kwargs:
            raise ValueError(
                "reasoning_mode and chat_template_kwargs['enable_thinking'] both set "
                f"for model {self.name!r}; set exactly one so the request is unambiguous"
            )
        clashes = sorted(_RESERVED_EXTRA_BODY_KEYS & set(self.request_extra_body))
        if clashes:
            raise ValueError(
                f"request_extra_body for model {self.name!r} may not set backend-owned "
                f"key(s) {clashes}; use the dedicated config fields instead"
            )
        return self

    def resolved_chat_template_kwargs(self) -> dict[str, Any]:
        """Chat-template kwargs actually sent: ``reasoning_mode`` plus explicit kwargs.

        The two sources are validated to be disjoint (see ``_no_silent_overwrites``),
        so the merge order below can never drop a model-declared value.
        """
        merged: dict[str, Any] = {}
        if self.reasoning_mode == "thinking":
            merged["enable_thinking"] = True
        elif self.reasoning_mode == "non-thinking":
            merged["enable_thinking"] = False
        merged.update(self.chat_template_kwargs)
        return merged

    def resolved_extra_body(self) -> dict[str, Any] | None:
        """The full ``extra_body`` for one chat-completion request (None if empty)."""
        body: dict[str, Any] = dict(self.request_extra_body)
        template_kwargs = self.resolved_chat_template_kwargs()
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        return body or None


# --------------------------------------------------------------------------- #
# Scaffolds
# --------------------------------------------------------------------------- #
class ScaffoldType(str, Enum):
    S1_REACT = "s1_react"
    S2_PLANNER = "s2_planner"
    S3_VERIFIER = "s3_verifier"


class ScaffoldConfig(_Strict):
    name: str
    kind: ScaffoldType
    max_steps: int = 15  # step budget = ToolsExecutionLoop max_iters
    notes_enabled: bool = False  # S2/S3: a writable note store (the M/H carriers)


# --------------------------------------------------------------------------- #
# Substrate / budget / margins / paths / seeds
# --------------------------------------------------------------------------- #
class SubstrateConfig(_Strict):
    id: Literal["A", "B"] = "A"  # A = MIRAGE-Dojo (AgentDojo), B = MIRAGE-Range (later)
    agentdojo_version: str = "v1.2.2"  # benchmark version string for get_suite
    suites: list[str] = Field(default_factory=lambda: ["banking", "slack", "travel", "workspace"])


class BudgetConfig(_Strict):
    max_steps: int = 15
    max_completion_tokens: int = 4096
    max_prompt_tokens: int | None = None
    max_tool_calls: int = 30
    max_retries: int = 3
    wall_clock_s: float = 600.0


class MarginsConfig(_Strict):
    """Equivalence margins and the measured epsilon floors.

    ``epsilon`` is populated by CV-0(a); before that it holds the design's
    placeholder assumptions. ``delta_*`` are the pre-registered equivalence
    margins (design 8.3), conditioned on CV-4's MDE.
    """

    delta_branch: float = 0.05
    delta_prog: float = 0.05
    epsilon: dict[str, float] = Field(default_factory=dict)


class PathsConfig(_Strict):
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    model_cache: Path = Path("models_cache")


class SeedConfig(_Strict):
    global_seed: int = 20260719
    # The per-branch seed = H(global_seed, checkpoint_id, branch_id, replicate);
    # this constant salts the hash so schedules can be regenerated independently.
    schedule_salt: str = "mirage-persist/cv0"


# --------------------------------------------------------------------------- #
# CV-0 sub-experiment blocks
# --------------------------------------------------------------------------- #
class EligibilityRule(_Strict):
    """Design 4.2 eligible-checkpoint criteria (pre-removal quantities only)."""

    total_rollouts: int = 5
    min_reachable_rollouts: int = 3  # reachable in >= 3 of 5
    min_subgoals: int = 1  # competence floor: >= 1 subgoal satisfied
    min_remaining_steps: int = 3  # horizon-5 attainable
    min_viable_continuations: int = 2  # a real branch, not a forced move
    require_exact_twin_digest: bool = True


class DeterminismAuditConfig(_Strict):
    """CV-0(a)."""

    n_checkpoints: int = 200
    n_twin_restores: int = 2
    K: int = 16
    greedy_horizon: int = 10  # steps for the greedy digest-equality continuation
    stochastic_horizon: int = 10  # steps for the twin-split-null continuations
    branch_horizon_for_epsilon: int = 5  # the h in Y^branch_h used for the ep floor gate


class CapabilityScreenConfig(_Strict):
    """CV-0(b)."""

    rollouts_per_cell: int = 5
    max_checkpoints_per_rollout: int = 20  # cap on candidate branch points per rollout
    max_tasks_per_suite: int | None = None  # None => all tasks (bound smoke/HF runs)
    branch_probe_K: int = 8  # continuations sampled to assess "viable continuations"
    branch_probe_horizon: int = 3
    eligibility: EligibilityRule = Field(default_factory=EligibilityRule)


class PotencyScreenConfig(_Strict):
    """CV-0(c)."""

    intervention_classes: list[str] = Field(default_factory=list)  # registry names; empty = all registered
    n_checkpoints: int = 40
    n_exposures: int = 8


class VarianceEstimationConfig(_Strict):
    """CV-0(d)."""

    outcomes: list[str] = Field(default_factory=lambda: ["Y_branch_5", "Y_prog_T"])
    method: Literal["reml", "mom"] = "reml"
    # K values to re-solve the power table over (design 5.0)
    power_table_K: list[int] = Field(default_factory=lambda: [8, 12, 16, 24])
    power_table_delta: list[float] = Field(default_factory=lambda: [0.10, 0.075, 0.05])


class GatesConfig(_Strict):
    """CV-0 PASS thresholds and KILL conditions (design CV-0 card)."""

    # PASS
    digest_equality_min: float = 0.95
    twin_null_branch_mean_max: float = 0.03
    twin_null_branch_p95_max: float = 0.08
    eligible_per_cell_min: int = 100
    potency_pp_above_ba_min: float = 0.30
    invalid_action_max: float = 0.15
    min_families_pass_capability: int = 3
    # KILL
    kill_digest_equality: float = 0.90
    kill_twin_null_mean: float = 0.10
    kill_min_families: int = 3


# --------------------------------------------------------------------------- #
# Top-level experiment configs
# --------------------------------------------------------------------------- #
class ExperimentConfig(_Strict):
    experiment_id: str
    seed: SeedConfig = Field(default_factory=SeedConfig)
    substrate: SubstrateConfig = Field(default_factory=SubstrateConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    run_to_terminal_fraction: float = 0.25
    margins: MarginsConfig = Field(default_factory=MarginsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: list[ModelBackendConfig] = Field(min_length=1)
    scaffolds: list[ScaffoldConfig]
    notes: str | None = None  # free-text label for the run


class CV0Config(ExperimentConfig):
    determinism_audit: DeterminismAuditConfig = Field(default_factory=DeterminismAuditConfig)
    capability_screen: CapabilityScreenConfig = Field(default_factory=CapabilityScreenConfig)
    potency_screen: PotencyScreenConfig = Field(default_factory=PotencyScreenConfig)
    variance_estimation: VarianceEstimationConfig = Field(default_factory=VarianceEstimationConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)


__all__ = [
    "JsonScalar",
    "BackendType",
    "SamplingConfig",
    "ModelBackendConfig",
    "ScaffoldType",
    "ScaffoldConfig",
    "SubstrateConfig",
    "BudgetConfig",
    "MarginsConfig",
    "PathsConfig",
    "SeedConfig",
    "EligibilityRule",
    "DeterminismAuditConfig",
    "CapabilityScreenConfig",
    "PotencyScreenConfig",
    "VarianceEstimationConfig",
    "GatesConfig",
    "ExperimentConfig",
    "CV0Config",
]
