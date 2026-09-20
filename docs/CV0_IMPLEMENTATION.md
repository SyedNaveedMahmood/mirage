# CV-0 Implementation

## Purpose

CV-0 is the calibration stage for MIRAGE-Persist. It establishes checkpoint fidelity,
measures the reproducibility floor used by later Post-Removal Effect analyses, identifies
eligible checkpoints and capable model/scaffold cells, screens intervention potency,
and estimates variance components for power calculations. The scientific specification
and gate thresholds are defined in
[MIRAGE-Persist_Experimental_Design.md](MIRAGE-Persist_Experimental_Design.md).

This page describes the code as implemented, not certification that it fully
realizes the design. Review the implementation limitations described below
before treating results as confirmatory. The same limitations apply to every
model size and were not silently fixed in the Qwen3.5 extension.

## Components

- **Determinism audit:** compares greedy twin continuations by trajectory digest and
  measures twin-split null distributions from stochastic continuations.
- **Capability screen:** uses same-step reachability counts, first-rollout terminal
  subgoal success, remaining steps, distinct probe digests, greedy twin equality and
  invalid-tool fractions as eligibility proxies. No post-removal selection occurs.
- **Potency screen:** measures terminal security success during exposed rollouts
  relative to the benign-artifact control, on the first model/scaffold only.
  This is not a first-action or post-removal measurement.
- **Variance estimation:** fits between- and within-checkpoint variance components
  of untreated outcome samples and recomputes the power table. These are not
  directly measured paired treatment-effect variances.
- **Gate evaluation and reporting:** applies configured `PASS`/`FAIL`/`KILL` rules and
  writes both machine-readable and human-readable summaries.

## Repository mapping

| Area | Location |
|---|---|
| Configuration schema, loading, and hashing | `src/mirage_persist/config/` |
| AgentDojo adapter and checkpoint engine | `src/mirage_persist/substrate/dojo/` |
| Model backends and scaffolds | `src/mirage_persist/models/`, `src/mirage_persist/scaffolds/` |
| Interventions and carrier registry | `src/mirage_persist/interventions/`, `src/mirage_persist/resets/` |
| Deterministic outcomes and event records | `src/mirage_persist/outcomes/`, `src/mirage_persist/events/` |
| Statistical estimators | `src/mirage_persist/stats/` |
| CV-0 stages, gates, and report | `src/mirage_persist/experiments/cv0/` |
| Run provenance and artifact lifecycle | `src/mirage_persist/run/` |

## Execution flow

1. The CLI resolves YAML inheritance and references, applies dotted overrides, and
   validates the resulting configuration.
2. `RunContext` creates the run directory and starts provenance, terminal, seed, and
   GPU accounting.
3. The configured AgentDojo adapter, model backends, and scaffolds collect greedy
   checkpoints by round-robin task rollouts across model/scaffold/suite strata,
   stopping at the configured total cap (not 100 per cell).
4. The determinism, capability, potency, and variance stages run in sequence.
5. Gate results are aggregated into the overall verdict and all artifacts are written.
6. Selected summary artifacts are mirrored under `artifacts/<experiment_id>/`.
   These are overwritten by a later run of the same experiment; run directories
   are the authoritative provenance records.

## Code walkthrough and contracts

All paths in this section are relative to `src/mirage_persist/`.

### Entry points and configuration

`cli.py` provides `version`, `check-substrate` and `preflight` and registers
the experiment CLI from `experiments/cv0/cli.py`. Actual experiment commands
are `mirage cv0 run`, `mirage smoke mock` and `mirage smoke hf`; separate
`cv0 determinism/capability/potency/variance` CLI commands are not implemented.
`preflight` probes a live server and is not the offline configuration validator.

`config/loader.py` recursively resolves `extends`, deep-merges mappings,
replaces lists, inlines referenced model/scaffold YAML, applies dotted `-o`
overrides, then validates `CV0Config` from `config/schema.py`. Paths are resolved
relative to the referring file before the config-root fallback. Models reject
unknown keys and ambiguous thinking/chat-template settings. Cyclic inheritance
is rejected. `config/hashing.py` hashes the resolved configuration, excluding
output paths and notes, not the YAML text.

The Qwen chain is:

```text
configs/base.yaml
  -> configs/cv0/cv0_full.yaml
     -> cv0_cluster_single_family_100cp.yaml
        -> cv0_qwen3p5_{4b,9b,27b}_100cp.yaml
           + referenced configs/models/<model>.yaml
```

The shared single-family base intentionally has `models: []` and cannot run
alone. Each child supplies one model and a distinct experiment ID. All retain
the same screens/seeds/scaffolds/budgets; only the family-count gates are relaxed
from three to one in the shared base. `cv0_full.yaml` itself still has only an
example model slot and is not a configured multi-family confirmation.

### Backend, scaffold and substrate boundary

`models/base.py::build_backend` selects mock, local Hugging Face or
OpenAI-compatible execution. Backends create AgentDojo-compatible LLM pipeline
elements; `scaffolds/base.py` supplies the initial system prompt/state.
S1 uses the substrate prompt. S2 appends a planner instruction and initializes
an empty plan/notes object, but that external object is not actively maintained
by the loop. S3 currently reuses the planner implementation.

`models/seeded_openai.py::SeededOpenAILLM` converts AgentDojo messages and tool
schemas, submits model/seed/temperature/top-p/max-tokens/request extras and
records returned usage in `extra_args`. `OpenAICompatBackend` reads URL/key
from the configured environment names. It does not load weights: vLLM runs in
a separate process/environment. Request seeds do not guarantee GPU determinism;
CV-0(a) measures it. BadRequest/422 responses are not retried by the outer
retry wrapper; other errors may be retried. Runtime retry accounting is not
driven by the full configured budget meter.

`substrate/dojo/adapter.py` loads AgentDojo task suites, initializes task/pre-task
environments, exposes tools/ground truth and reconstructs traces. The pinned
AgentDojo source SHA is separate from benchmark version `v1.2.2`. Model decisions
and tool execution remain on AgentDojo interfaces; no substrate fork is needed.

### Checkpoints, continuations and outcomes

`substrate/dojo/agent_loop.py::run_agent_loop` checks budget, calls a pre-decision
hook, derives a per-step seed, requests an LLM decision, accounts for usage and
executes tools. A decision without tool calls is terminal. Invalid actions are
tool-result errors, not a general judge of prose quality. Token/tool limits
are checked between decisions, so a multi-call decision may cross a limit.

`continuation.py::rollout` captures checkpoints through that hook.
`checkpoint.py::Checkpoint` stores message history, current and pre-task
environments, task/step identity, scaffold state, injections and remaining
steps. Restores deep-copy mutable state; `digest.py` canonicalizes it for
comparison. `continue_branch` restores history/environments, creates a fresh
LLM element and fresh continuation budget, then runs the chosen horizon.
It does not restore an opaque vLLM KV cache or consumed token/time budget.

`run/seeds.py` hashes global seed, salt, checkpoint ID, branch ID and replicate,
then derives decision seeds. Different branch IDs yield different seeds;
despite some historical CRN wording, this is not shared random numbers across
arbitrarily named branches. Each Qwen size also generates its own checkpoints.

`outcomes/scoring.py` extracts post-checkpoint tool calls, pairs them with tool
errors and dispatches deterministic AgentDojo utility/security predicates
(trace-based predicates take priority). `Y_branch_h` tests for a valid
ground-truth call within the first h post-checkpoint decisions, not an LLM
judgement. Progress uses task subgoal definitions/fallbacks in `outcomes/subgoals.py`.
Action-class distributions come from `outcomes/action_class.py`.

### CV-0 stages and reporting

`experiments/cv0/report.py::run_cv0` builds the adapter/backends/scaffolds,
records model provenance, opens `events.jsonl` and executes:

1. `common.py::collect_checkpoints`, followed by `determinism_audit.py`:
   greedy twin digests and two groups of K stochastic continuations per checkpoint.
   Null summaries and pooled untreated outcomes feed later stages.
2. `capability_screen.py`: its own stochastic rollout/probe census across
   model/scaffold cells and suites, not a filter applied to the earlier
   determinism sample.
3. `potency_screen.py`: registered injection templates from `interventions/`
   applied during full exposure rollouts of the primary cell; no removal stage.
4. `variance_estimation.py`: checkpoint-level variance via REML with fallback
   behavior in `stats/variance_components.py`, and `stats/power.py` power tables.
5. `gates.py`: digest equality, branch null floor, eligible counts, potency and
   invalid-action family floor. `experiments/base.py` aggregates KILL before
   FAIL before PASS. There is no separate variance-estimation PASS gate.

The CLI maps overall PASS/FAIL/KILL to exit codes 0/1/2. Gates are evaluated
after all stages, not as early-stopping criteria between screens. There is no
stage-resume API. An infrastructure exception can prevent final summaries.

### Run and cluster lifecycle

`run/run_context.py::RunContext` creates a timestamp/config-hash directory,
captures terminal output, seeds local RNGs, writes resolved YAML/package freeze
and records code/substrate/model/GPU provenance. Normal exceptions are recorded
on exit; a scheduler hard kill can prevent final manifest writing. Substrate SHA
mismatch is currently a warning, not an automatic abort. Out-of-process GPU-hours
are a wall-clock proxy, not measured kernel utilization.

Outside `src/`, `cluster/bwunicluster3/submit_cv0_model.sh` validates a profile's
configuration, arranges optional prefetch and submits `cv0_model.sbatch`.
The batch job validates again, allocates a local port, launches the profile's
vLLM command, checks health/model identity, writes job provenance, performs API
preflight, runs CV-0 and cleans up. `MIRAGE_JOB_MODE=smoke` exits after preflight.
The Qwen sequence wrapper composes this job lifecycle through Slurm
dependencies; it does not implement its own model execution loop.

## Configuration

`configs/base.yaml` defines shared defaults. Important runnable configurations include:

- `configs/smoke/cv0_smoke_mock.yaml` for fast, GPU-free engine validation;
- `configs/smoke/cv0_smoke_hf.yaml` for a small local-model validation;
- `configs/cv0/cv0_full.yaml` for the full calibration shape;
- `configs/cv0/cv0_cluster_single_family_100cp.yaml` as the shared base for
  single-family cluster runs;
- `configs/cv0/cv0_<profile>_100cp.yaml` for pinned production model arms,
  including `qwen3p5_4b`, `qwen3p5_9b` and `qwen3p5_27b` (one family).

Model and scaffold entries can reference separate YAML files. Command-line overrides
use dotted keys, for example:

```powershell
mirage cv0 run --config configs/cv0/cv0_full.yaml `
  -o determinism_audit.n_checkpoints=50 -o determinism_audit.K=16
```

## Artifacts

Each run may produce:

- `envelope.json`: digest equality and outcome-specific reproducibility floors;
- `figure_s1_twin_split_null.png`: twin-split null distributions;
- `eligibility.parquet`: checkpoint eligibility census;
- `capability_report.md`: cell counts, invalid-action rates, and gate summary;
- `potency.json`: immediate diversion results by intervention class;
- `variance_power.json`: variance components and recalculated power table;
- `cv0_verdict.json`: per-gate and overall verdict;
- `events.jsonl`: append-only checkpoint/continuation event records;
- `manifest.json`, `resolved_config.yaml`, `env_freeze.txt`, and `terminal.log`: run
  provenance and diagnostics.

The mirror currently contains the envelope, figure, eligibility, capability
report and verdict; `potency.json` and `variance_power.json` stay in the run
directory. Cluster logs live separately under `runs/_cluster_jobs/` and are
also copied into the selected run directory after the experiment. Avoid concurrent
reruns of the same experiment ID: that copy step finds the latest directory.

## Validation

`mirage smoke mock` runs all four stages with a deterministic mock backend and requires
byte-exact checkpoint behavior. `mirage smoke hf` exercises seeded generation, tool-call
parsing, token accounting, and provenance with a small Hugging Face model. Unit tests
cover configuration, checkpoint restoration, continuations, interventions, scoring,
statistics, provenance, cluster helpers, and the end-to-end mock path.

## Full-scale execution

Smoke configurations validate wiring and invariants at small sample sizes; their
scale-dependent thresholds do not constitute a production CV-0 verdict. Full-scale
runs use the registered model families, K=16 continuation sampling for the determinism
envelope, and the production thresholds in the CV-0 configuration. Cluster setup,
prefetch, submission, monitoring, and result collection are documented in
[`cluster/bwunicluster3/README.md`](../cluster/bwunicluster3/README.md) and
[`cluster/bwunicluster3/CV0_RUNBOOK.md`](../cluster/bwunicluster3/CV0_RUNBOOK.md).
