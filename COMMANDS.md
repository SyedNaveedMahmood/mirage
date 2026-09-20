# COMMANDS — running CV-0

All commands assume Windows PowerShell from the repo root. The venv Python is
`.\.venv\Scripts\python.exe`; after activation you can use `mirage` / `pytest` directly.
Every `mirage` run also tees its terminal output to the run directory's `terminal.log`.

## 0. One-time setup

```powershell
# Creates a venv that REUSES the system GPU stack (torch/transformers), then installs
# the pinned AgentDojo substrate + MIRAGE (+ hf, dev extras).
.\scripts\setup_env.ps1

# ...or manually:
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
pip install -e prior_work_repos\agentdojo         # pinned SHA 089ed468, v0.1.35
pip install -e ".[hf,dev]"
```

Optional (only for the real-model HF smoke; downloads ~3 GB):

```powershell
.\scripts\fetch_smoke_model.ps1                    # Qwen/Qwen2.5-1.5B-Instruct
```

Optional (only for the fleet openai-compatible path — a vLLM/llama.cpp/LM Studio/Ollama
server): set the endpoint in the environment (no secrets in configs):

```powershell
$env:OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_COMPATIBLE_API_KEY  = "EMPTY"
```

## 1. Sanity checks

```powershell
mirage version                                     # version + pinned substrate SHA
mirage check-substrate --suite banking             # AgentDojo's own suite self-check
mirage check-substrate --suite slack --version v1.2.2
pytest -q                                           # 34 tests
pytest -q -m "not slow"                             # skip the end-to-end CV-0 test
ruff check src tests
```

## 2. Smoke tests (dev 2060)

```powershell
mirage smoke mock          # GPU-free, byte-exact engine + determinism (~6 s). MUST be PASS.
mirage smoke hf            # tiny real model on the GPU; measures the REAL decoder envelope
.\scripts\run_cv0_smoke.ps1            # runs mock (add --hf to also run hf), with a transcript
```

## 3. CV-0 (full and per-part)

```powershell
# Full CV-0: all four sub-experiments + gates + all artifacts.
mirage cv0 run --config configs\cv0\cv0_full.yaml

# Override any config knob on the command line (dotted path, repeatable):
mirage cv0 run --config configs\cv0\cv0_full.yaml `
  -o determinism_audit.n_checkpoints=50 -o determinism_audit.K=16

# A smoke-scale full run using the mock backend:
mirage cv0 run --config configs\smoke\cv0_smoke_mock.yaml
```

NOTE: `configs\cv0\cv0_full.yaml` is the FLEET config (100 checkpoints, K=16, 4 suites). It
does not fit the 12 GB dev 2060 — run it on the 4080S/4090/3090/H100 fleet, replacing the
single `openai-compatible` model slot with the real M1..M4 families. On the dev box use the
smoke configs or `-o` overrides to shrink N.

## 4. Where outputs land

Each run creates `runs\<experiment_id>\<UTC-timestamp>_<config_hash8>\`:

| file | contents |
|---|---|
| `manifest.json` | full provenance (time, GPU name/hours/VRAM, git SHAs, config hash, model, tokenizer, seeds, deps, exit status) |
| `terminal.log` | tee of all stdout/stderr for this run (for debugging) |
| `resolved_config.yaml` | the exact merged config used |
| `env_freeze.txt` | pip freeze snapshot (+ sha256 in the manifest) |
| `events.jsonl` | per-(checkpoint,branch,step) event stream |
| `envelope.json` | epsilon floors + digest-equality rate (CV-0a) |
| `figure_s1_twin_split_null.png` | Figure S1 — twin-split null distributions |
| `eligibility.parquet` | eligible-checkpoint census (CV-0b) |
| `capability_report.md` | human-readable capability + gate summary |
| `potency.json` | per-class immediate diversion vs BA (CV-0c) |
| `variance_power.json` | fitted sigma_tau + re-solved power table (CV-0d) |
| `cv0_verdict.json` | PASS/FAIL/KILL per gate + overall |

The four canonical artifacts (`envelope.json`, `eligibility.parquet`, `capability_report.md`,
Figure S1) and `cv0_verdict.json` are also mirrored to `artifacts\<experiment_id>\`.

## 5. Terminal capture (belt-and-suspenders)

The CLI already tees to `terminal.log`. For an additional OS-level transcript:

```powershell
Start-Transcript -Path "runs\_transcripts\cv0_$(Get-Date -Format yyyyMMdd_HHmmss).txt"
mirage cv0 run --config configs\cv0\cv0_full.yaml
Stop-Transcript
```

## 6. Exit codes

`mirage cv0 run` / `mirage smoke *` exit **0 = PASS, 1 = FAIL, 2 = KILL** so CI can gate on
the CV-0 verdict.

## 7. bwUniCluster 3.0 (Qwen3.5 sizes and other model families)

The cluster profile uses two isolated environments and one pinned vLLM server per
model. From the directory that contains `llmrun/`:

```bash
cd llmrun
mkdir -p agentrun
git clone --branch main https://github.com/SyedNaveedMahmood/mirage.git agentrun/mirage-persist
mkdir -p agentrun/vllm
cd agentrun/mirage-persist

bash cluster/bwunicluster3/submit_setup.sh
# Wait for the setup job to finish, then per model:
bash cluster/bwunicluster3/submit_prefetch_model.sh gemma4_26b_a4b   # CPU, exact revision
bash cluster/bwunicluster3/submit_smoke_model.sh   gemma4_26b_a4b    # optional, dev queue, <30 min
bash cluster/bwunicluster3/submit_cv0_model.sh     gemma4_26b_a4b    # production run
```

Profiles: `qwen3p5_4b`, `qwen3p5_9b`, `qwen3p5_27b`, `gemma4_26b_a4b`,
`gpt_oss_20b`, `llama31_8b`.
`bash cluster/bwunicluster3/submit_cv0.sh` still works and is equivalent to
`submit_cv0_model.sh qwen3p5_9b`.

### Qwen3.5 size sequence

```bash
# Offline config/profile checks only: no Slurm or model calls.
bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --dry-run

# Submit separate, sequential GPU jobs: 4B -> 9B -> 27B.
SBATCH_GPU_PARTITION=gpu_h100 bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh

# Optional: cancel successors on a preceding nonzero exit (including FAIL/KILL).
SBATCH_GPU_PARTITION=gpu_h100 bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --stop-on-failure
```

Default behavior attempts all three after the previous GPU job finishes,
regardless of verdict; model prefetch must always succeed. `CV0_CONFIG` must be
unset. The 27B profile is H100-only and its 47-hour budget is not a measured
runtime. See the [sequence runbook](cluster/bwunicluster3/QWEN35_SEQUENCE.md)
for smoke checks, ledgers, cancellation and results, and the
[code walkthrough](docs/CV0_IMPLEMENTATION.md) for implementation limitations.

Useful overrides:

```bash
SBATCH_GPU_PARTITION=gpu_h100_il SBATCH_GPU_TIME=32:00:00   bash cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b
SBATCH_ACCOUNT=my_project SKIP_MODEL_PREFETCH=1   bash cluster/bwunicluster3/submit_cv0_model.sh llama31_8b
```

See `cluster/bwunicluster3/README.md` and `cluster/bwunicluster3/CV0_RUNBOOK.md`
for resource requests, partition selection, gated-model access, logs and result
paths.

## 8. Server preflight

Cheap functional check of a running OpenAI-compatible server (a few dozen tokens):
`/v1/models`, one ordinary completion, and one request with a JSON tool schema
whose response must parse back as a tool call.

```bash
mirage preflight --config configs/cv0/cv0_llama31_8b_100cp.yaml   --json-out runs/_preflight/llama31.json
```

Exit code 0 means every check passed. The cluster job runs this automatically
before starting CV-0.
