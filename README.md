# MIRAGE-Persist

**Counterfactual measurement of persistent policy hysteresis in deception-aware AI agents.**

MIRAGE-Persist measures whether a temporary deception, once *removed*, still changes an
agent's future behaviour (the **Post-Removal Effect**), and decomposes that effect across
recorded state carriers. This repository currently implements **CV-0** — the calibration
instrument that must PASS before any other experiment starts:

- exact checkpoint restore over an [AgentDojo](https://github.com/ethz-spylab/agentdojo)-derived
  substrate (`MIRAGE-Dojo`, Substrate A);
- the reproducibility/sensitivity floor (the ε every later effect is compared against);
- the eligible-checkpoint census, model-capability screen, and intervention-potency screen;
- the variance components that re-solve the sample-size table.

See [docs/](docs/) for the proposal and full experimental design,
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the engineering guide, and
[COMMANDS.md](COMMANDS.md) for the command reference.

The [CV-0 code walkthrough](docs/CV0_IMPLEMENTATION.md) documents execution,
data contracts, artifacts and current implementation limitations.

## Status

CV-0 (four sub-experiments + gates + artifacts) over Substrate A, all four AgentDojo
suites, scaffolds S1/S2. Backends: mock (byte-exact, GPU-free), HuggingFace in-process
(tiny real models), and any OpenAI-compatible server (vLLM/llama.cpp/LM Studio/Ollama).

Qwen3.5 **4B, 9B and 27B** now have pinned single-model cluster configurations
and a sequential launcher. This is a within-family size comparison, not a
three-family confirmation. Offline validation does not establish GPU memory fit
or a CV-0 PASS; 4B/27B still need cluster smoke tests. S2 notes and several
scientific-screen approximations are documented in the code walkthrough, not treated as
validated implementations of the full experimental design.

## Quickstart

```powershell
# 1. venv that reuses the system GPU stack (torch/transformers), then install deps
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
git clone https://github.com/ethz-spylab/agentdojo.git prior_work_repos/agentdojo
git -C prior_work_repos/agentdojo checkout --detach 089ed468cf3ed0322acc66b0211f26d9d90dbf60
pip install -e prior_work_repos\agentdojo      # pinned substrate (SHA 089ed468, v0.1.35)
pip install -e ".[hf,dev]"

# 2. sanity checks
mirage version
mirage check-substrate --suite banking
pytest -q

# 3. smoke (GPU-free, byte-exact engine + determinism)
mirage smoke mock
```

Full commands are in [COMMANDS.md](COMMANDS.md).

## bwUniCluster 3.0

The [cluster guide](cluster/bwunicluster3/README.md) creates independent vLLM
and MIRAGE environments under `llmrun/agentrun/`. Clone this repository as
`agentrun/mirage-persist` from `https://github.com/SyedNaveedMahmood/mirage.git`.

After setup, preview or submit the Qwen size sequence from Bash:

```bash
# Offline only: validates all three configurations; submits nothing.
bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --dry-run

# When ready: separate GPU allocations, 4B -> 9B -> 27B.
SBATCH_GPU_PARTITION=gpu_h100 \
  bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh
```

The default attempts every size even after an earlier FAIL/KILL; add
`--stop-on-failure` to require each predecessor to PASS. CPU prefetches may
overlap, GPU runs do not. See the [sequence runbook](cluster/bwunicluster3/QWEN35_SEQUENCE.md)
for short smoke checks, resource limits, job ledgers, cancellation and results.

## Design invariants

- **AgentDojo is a pinned dependency, never forked.** We subclass its pipeline elements
  and drive its suites; seeds ride the `extra_args` carrier.
- **Every run is fully provenanced** (`runs/<exp>/<ts>_<hash>/manifest.json` + `terminal.log`).
- **Experiments are mutated via YAML configs**, not code edits.
- **No LLM judge for any confirmatory outcome** — all CV-0 outcomes are deterministic.
