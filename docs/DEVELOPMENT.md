# Development Guide

This guide summarizes the repository-level engineering conventions for
MIRAGE-Persist. Scientific definitions and experiment specifications are maintained in
[MIRAGE-Persist_Experimental_Design.md](MIRAGE-Persist_Experimental_Design.md), while
the full command reference is in [`COMMANDS.md`](../COMMANDS.md).

For implementation-level contracts and execution flow, see the
[CV-0 code walkthrough](CV0_IMPLEMENTATION.md), including current scientific
approximations. Hardware validation guidance is in the
[Qwen3.5 sequence runbook](../cluster/bwunicluster3/QWEN35_SEQUENCE.md).

## Architecture

The package uses a top-down import structure:

```text
config/          Strict Pydantic schemas, YAML inheritance/references, overrides, hashing
run/             Run lifecycle, provenance, terminal capture, seed schedule, GPU metering
substrate/dojo/  AgentDojo adapter, state digests, checkpoints, budgets, agent loop, continuations
models/          Mock, local Hugging Face, and OpenAI-compatible backends
scaffolds/       S1 ReAct and S2 planner prompt (external notes not actively maintained)
interventions/   Versioned intervention templates and structural removal operators
resets/          Carrier registry and reset operators
outcomes/        Component-wise deterministic scoring, action classes, subgoals
events/          Append-only JSONL event schema and writer
stats/           Null estimation, bootstrap, variance components, power, divergences
experiments/cv0/ CV-0 drivers, gates, report generation, and CLI
analysis/        Figure and artifact I/O helpers
cli.py           `mirage` command entry point
```

## AgentDojo integration

AgentDojo is treated as a pinned external dependency. The tested revision is v0.1.35
at commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60`; every run records the installed
revision and reports a mismatch. MIRAGE extends AgentDojo pipeline interfaces rather
than maintaining a fork. The benchmark version passed to `get_suite()` is configured
separately.

Per-decision sampling metadata uses AgentDojo's existing `extra_args` carrier:

- `mirage_seed` selects the continuation seed;
- `mirage_greedy` enables greedy decoding for determinism checks;
- `mirage_usage` accumulates prompt and completion token counts.

The only AgentDojo boundary is `substrate/dojo/adapter.py`. Substrate API changes should
be isolated there where possible.

## Checkpoints and continuations

Checkpoints are captured immediately before an agent decision, after the initial user
message or a tool observation. A continuation therefore re-makes the branch decision
instead of resuming after it. Environment state uses `model_copy(deep=True)`, and
messages and scaffold state are deep-copied for every restore so sibling branches do
not share mutable state.

Intervention removal is structural: the untreated environment is rebuilt from
AgentDojo defaults instead of editing injected strings in place. Removal verification
checks the resulting state digest and absence of the injected text.

The checkpoint is the independent statistical unit. Replicated continuations from one
checkpoint are dependent samples; interval estimation and bootstrap procedures cluster
over checkpoints.

## Configuration workflow

Experiment behavior is controlled through YAML under `configs/`. The loader supports
`extends`, model/scaffold file references, and dotted CLI overrides. Schemas use
`extra="forbid"`, so unknown keys fail validation instead of being ignored. New
experiment settings belong in the schema and configuration files rather than as
hard-coded evaluator branches.

## Run provenance

Every experiment runs inside `RunContext` and writes to:

```text
runs/<experiment_id>/<UTC timestamp>_<config hash>/
```

The directory contains the resolved configuration, environment freeze, terminal log,
event stream, manifest, and experiment artifacts. The manifest records code and
substrate revisions, model and tokenizer identity, sampling settings, seed schedule,
hardware and GPU use, configuration hash, artifacts, warnings, and the final gate
verdict. Text artifacts are UTF-8; CLI startup also enables UTF-8 console handling on
Windows.

## Extension points

- **Backend:** implement `models.base.Backend` and add dispatch in `build_backend`.
- **Intervention:** register a versioned `Intervention` in
  `interventions/registry.py`, including a removal operator with a verifiable
  postcondition.
- **Carrier reset:** implement `ResetOperator.reset(state, untreated)` in
  `resets/registry.py`. The untreated state provides the paired reset-artifact control.
- **Task subgoals:** register deterministic predicates through
  `outcomes.subgoals.register_subgoals`; unregistered tasks fall back to AgentDojo
  utility as one binary subgoal.
- **Experiment:** reuse the substrate, outcome, statistics, event, and run layers. The
  six-arm branch taxonomy is defined by `continuation.BranchLabel`.

## CV-0 gates

`experiments/cv0/gates.py` emits `PASS`, `FAIL`, or `KILL` for each gate. The overall
status is `KILL` if any gate is `KILL`, otherwise `FAIL` if any gate is `FAIL`, and
`PASS` otherwise. CLI exit codes are 0, 1, and 2 respectively. Thresholds are defined
in the validated experiment configuration. Smoke configurations use explicit reduced-scale
thresholds and do not constitute a production verdict.

## Development commands

```powershell
mirage version
mirage check-substrate --suite banking
pytest -q
ruff check src tests
mirage smoke mock
mirage smoke hf
mirage cv0 run --config configs/cv0/cv0_full.yaml
```

The mock smoke is GPU-free and verifies byte-exact engine behavior. The Hugging Face
smoke exercises a real local decoder. Full-scale CV-0 runs use the model profiles and
shared Slurm bodies documented in [`cluster/bwunicluster3/README.md`](../cluster/bwunicluster3/README.md).
