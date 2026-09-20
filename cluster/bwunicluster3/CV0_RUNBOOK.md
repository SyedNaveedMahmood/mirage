# Running CV-0 on bwUniCluster 3.0

For the Qwen3.5 **4B -> 9B -> 27B** sequential workflow, use the dedicated
[sequence runbook](QWEN35_SEQUENCE.md) and read the
[code walkthrough](../../docs/CV0_IMPLEMENTATION.md). The sections below also retain
the older per-model and interactive workflows.

This runbook covers three workflows:

1. the per-model production CV-0 campaign (Qwen3.5-9B, Gemma 4 26B-A4B,
   gpt-oss-20b, Llama-3.1-8B-Instruct) -- see
   [Multi-model CV-0 campaign](#multi-model-cv-0-campaign);
2. a short per-model smoke test on the development GPU queue; and
3. a timed, reduced CV-0 run in an interactive 30-minute development allocation.

The cluster login shell is Bash. PowerShell constructs such as
`[Diagnostics.Stopwatch]`, `$timer`, backtick line continuations, and Windows
path separators do not work there. The Bash equivalent uses `time`, `\` line
continuations, and `/` path separators.

## Prerequisites

The expected directory layout is:

```text
llmrun/
└── agentrun/
    ├── vllm/
    └── mirage-persist/
```

Log in to the cluster and enter the repository:

```bash
cd llmrun/agentrun/mirage-persist
```

Confirm that you are on `main` of `SyedNaveedMahmood/mirage`:

```bash
git branch --show-current
```

Expected output:

```text
main
```

### Create the environments once

If the two environments have not been created yet, submit the setup job:

```bash
bash cluster/bwunicluster3/submit_setup.sh
```

Monitor it and wait for it to finish successfully:

```bash
squeue -u "$USER"
tail -f cluster/bwunicluster3/logs/mirage-setup-*.out
```

Verify the resulting executables:

```bash
test -x .venv/bin/mirage && echo "MIRAGE environment: OK"
test -x ../vllm/.venv/bin/vllm && echo "vLLM environment: OK"
```

### Prefetch Qwen3.5-9B once

Download the pinned model snapshot before requesting a 30-minute GPU. Model
download time should not consume the short GPU allocation.

```bash
mkdir -p cluster/bwunicluster3/logs

sbatch \
  --partition=cpu \
  --chdir="$PWD" \
  cluster/bwunicluster3/prefetch_qwen3p5_9b.sbatch
```

Wait for the prefetch job to finish and check its log:

```bash
squeue -u "$USER"
tail -f cluster/bwunicluster3/logs/qwen35-fetch-*.out
```

## Timed reduced run on a 30-minute development allocation

This workflow tests an actual Qwen3.5-9B inference path. The 30-minute Slurm
limit includes model loading, CUDA/Triton initialization and compilation, vLLM
startup, and the MIRAGE run. The `time` measurement below measures only the
MIRAGE command, so it will be shorter than the total Slurm allocation time.

The development queue is for debugging and performance testing. It allows one
running development job per user and has a 30-minute maximum.

### 1. Request an interactive H100

From `llmrun/agentrun/mirage-persist`, run:

```bash
salloc \
  --partition=dev_gpu_h100 \
  --time=00:30:00 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G
```

Wait until Slurm grants the allocation. Then verify the GPU:

```bash
hostname
nvidia-smi
```

Do not run the following GPU commands on the login node.

### 2. Start vLLM on the allocated node

```bash
VLLM_ROOT="$(cd ../vllm && pwd)"

export HF_HOME="$VLLM_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$VLLM_ROOT/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$VLLM_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$VLLM_ROOT/cache/triton"
export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

mkdir -p \
  "$HF_HUB_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_ROOT/logs"

"$VLLM_ROOT/.venv/bin/vllm" serve "Qwen/Qwen3.5-9B" \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name "Qwen/Qwen3.5-9B" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key EMPTY \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --download-dir "$HF_HUB_CACHE" \
  > "$VLLM_ROOT/logs/vllm-smoke.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
```

### 3. Wait for vLLM readiness

```bash
until curl --silent --fail \
  -H "Authorization: Bearer EMPTY" \
  http://127.0.0.1:8000/health >/dev/null
do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM failed during startup"
    tail -n 100 "$VLLM_ROOT/logs/vllm-smoke.log"
    break
  fi
  echo "Waiting for vLLM..."
  sleep 5
done
```

Verify that the expected served model is visible:

```bash
curl --silent --fail \
  -H "Authorization: Bearer EMPTY" \
  http://127.0.0.1:8000/v1/models
```

If readiness fails, inspect:

```bash
tail -n 100 "$VLLM_ROOT/logs/vllm-smoke.log"
```

### 4. Run the timed reduced command

This is the Bash translation of the requested PowerShell command:

```bash
time ./.venv/bin/mirage cv0 run \
  --config configs/cv0/cv0_full.yaml \
  -o determinism_audit.n_checkpoints=10 \
  -o determinism_audit.K=2 \
  -o capability_screen.rollouts_per_cell=1 \
  -o capability_screen.max_tasks_per_suite=2 \
  -o capability_screen.branch_probe_K=2 \
  -o capability_screen.branch_probe_horizon=2 \
  -o potency_screen.n_checkpoints=4 \
  -o potency_screen.n_exposures=1 \
  -o gates.eligible_per_cell_min=1 \
  -o gates.min_families_pass_capability=1 \
  -o gates.kill_min_families=1
```

Bash prints three measurements when the command finishes:

```text
real    total wall-clock time
user    CPU time spent in user code
sys     CPU time spent in kernel code
```

This command is reduced relative to the full suite, but it still inherits four
suites, two scaffolds, ten-step determinism horizons, and up to 20 capability
checkpoints per rollout from `cv0_full.yaml`. It is therefore not guaranteed to
finish within 30 minutes. If Slurm reaches the limit, it terminates the allocation.
The partial terminal and vLLM logs can still confirm how far startup and execution
progressed.

### 5. Stop vLLM and release the allocation

After the run finishes or fails:

```bash
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
exit
```

## Full CV-0 suite

Run the full suite as a non-interactive batch job. Do not run it in a development
queue or directly on a login node.

### 1. Enter the repository

```bash
cd llmrun/agentrun/mirage-persist
```

### 2. Confirm the production time limit

```bash
grep '^#SBATCH --time=' cluster/bwunicluster3/cv0_model.sbatch
grep 'SBATCH_GPU_TIME_DEFAULT' cluster/bwunicluster3/model_profiles/*.sh
```

The submission helper always passes the profile walltime explicitly, so the
per-profile default is what matters. All defaults are below 48 hours, which keeps
the job eligible for both the standard and the Ice Lake GPU queues.

### 3. Submit the suite

The default submission performs a CPU prefetch job first (only when the model is
not already cached at the pinned revision) and then submits the GPU job with an
`afterok` dependency:

```bash
bash cluster/bwunicluster3/submit_cv0_model.sh qwen3p5_9b
```

`bash cluster/bwunicluster3/submit_cv0.sh` remains as a compatibility wrapper for
exactly this command.

If the pinned model snapshot has already been prefetched successfully:

```bash
SKIP_MODEL_PREFETCH=1 \
bash cluster/bwunicluster3/submit_cv0.sh
```

If your allocation requires an account:

```bash
SBATCH_ACCOUNT=your_project_account \
bash cluster/bwunicluster3/submit_cv0.sh
```

The production job launches vLLM and runs:

```bash
./.venv/bin/mirage cv0 run \
  --config configs/cv0/cv0_bwunicluster_qwen3p5_9b.yaml
```

That cluster configuration executes CV-0(a-d) at the full configured scale for
the Qwen3.5-9B family. Its family-count gate is intentionally scoped to the one
model family in this allocation; it is not a replacement for the preregistered
three-family population run.

### 4. Monitor the jobs

The submission helper prints the model-prefetch and GPU job IDs. Monitor all of
your jobs with:

```bash
squeue -u "$USER"
```

Inspect Slurm output:

```bash
tail -f cluster/bwunicluster3/logs/cv0-qwen35-9b-*.out
```

Inspect the vLLM server log using the GPU job ID:

```bash
tail -f ../vllm/logs/vllm-JOB_ID.log
```

After completion, obtain accounting information:

```bash
sacct -j JOB_ID \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Timelimit,AllocTRES
```

### 5. Inspect results

Experiment outputs are written under:

```text
runs/cv0_bwunicluster_qwen3p5_9b/
```

Curated outputs are mirrored under:

```text
artifacts/cv0_bwunicluster_qwen3p5_9b/
```

Important files include:

- `manifest.json`: code, environment, model, GPU, seed, and config provenance;
- `terminal.log`: MIRAGE terminal output;
- `cv0_verdict.json`: overall and per-gate verdicts;
- `envelope.json`: determinism and twin-split envelope;
- `eligibility.parquet`: capability-screen checkpoint census;
- `potency.json`: intervention potency results; and
- `variance_power.json`: variance estimates and updated power table.

The MIRAGE exit codes are:

- `0`: PASS;
- `1`: FAIL; and
- `2`: KILL.

A Slurm job with exit code 1 or 2 may have completed correctly at the
infrastructure level while reporting a scientific FAIL or KILL. Always inspect
`cv0_verdict.json` and `terminal.log` before diagnosing it as a cluster failure.

## Useful cluster references

- [bwUniCluster 3.0 queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwUniCluster 3.0 GPU jobs](https://wiki.bwhpc.de/e/BwUniCluster3.0/Slurm#GPU_jobs)
- [Qwen3.5-9B model](https://huggingface.co/Qwen/Qwen3.5-9B)
- [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)

## Multi-model CV-0 campaign

Six model arms (four families) share one Slurm job body and one set of scientific settings
(100 determinism checkpoints, K=16, four suites, two scaffolds, identical seeds,
horizons and thresholds). Only the model, its vLLM flags and the experiment id
differ. Their generated checkpoints and assigned hardware may differ too;
matching configurations does not make these paired observations.

| Profile | Model | CV-0 config | Walltime default |
| --- | --- | --- | --- |
| `qwen3p5_4b` | `Qwen/Qwen3.5-4B` | `configs/cv0/cv0_qwen3p5_4b_100cp.yaml` | 22:00:00 |
| `qwen3p5_9b` | `Qwen/Qwen3.5-9B` | `configs/cv0/cv0_qwen3p5_9b_100cp.yaml` | 22:00:00 |
| `qwen3p5_27b` | `Qwen/Qwen3.5-27B` | `configs/cv0/cv0_qwen3p5_27b_100cp.yaml` | 47:00:00 |
| `gemma4_26b_a4b` | `google/gemma-4-26B-A4B-it` | `configs/cv0/cv0_gemma4_26b_a4b_100cp.yaml` | 30:00:00 |
| `gpt_oss_20b` | `openai/gpt-oss-20b` | `configs/cv0/cv0_gpt_oss_20b_100cp.yaml` | 30:00:00 |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | `configs/cv0/cv0_llama31_8b_100cp.yaml` | 22:00:00 |

All commands run from `llmrun/agentrun/mirage-persist`.

### 0. Accept the licences for the gated models

`google/gemma-4-26B-A4B-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated.
Accept the licence on each model page with your Hugging Face account, then
authenticate once on a login node with the same account:

```bash
export HF_HOME="$PWD/../vllm/cache/huggingface"
../vllm/.venv/bin/hf auth login          # or: export HF_TOKEN=<token>
```

The prefetch job fails with an explicit explanation when this has not been done.
Tokens are never written to any log.

### 1. Prefetch each model (CPU job)

```bash
bash cluster/bwunicluster3/submit_prefetch_model.sh gemma4_26b_a4b
bash cluster/bwunicluster3/submit_prefetch_model.sh gpt_oss_20b
bash cluster/bwunicluster3/submit_prefetch_model.sh llama31_8b
```

Only the pinned revision is downloaded. A completion marker keyed on model and
revision is written on success; the helper submits nothing when it already
exists. Force a re-download with `FORCE_MODEL_PREFETCH=1`.

For `llama31_8b` this also pins the Llama 3.1 JSON tool-use chat template into
`cluster/bwunicluster3/chat_templates/` — commit the template, its `.sha256` and
its `.provenance.json` afterwards. It can also be fetched by hand:

```bash
bash cluster/bwunicluster3/fetch_chat_template.sh llama31_8b
```

Follow each prefetch job:

```bash
squeue -u "$USER"
tail -f cluster/bwunicluster3/logs/fetch-gemma4_26b_a4b-*.out
```

### 2. Optional short smoke test (development queue, < 30 min)

```bash
bash cluster/bwunicluster3/submit_smoke_model.sh gemma4_26b_a4b
bash cluster/bwunicluster3/submit_smoke_model.sh gpt_oss_20b
bash cluster/bwunicluster3/submit_smoke_model.sh llama31_8b
```

Each smoke job loads the model, waits for `/health`, verifies `/v1/models`, runs
one ordinary generation and one tool call, then exits. It never runs the CV-0
campaign and never chains a follow-up job.

```bash
tail -f cluster/bwunicluster3/logs/smoke-gemma4_26b_a4b-*.out
```

### 3. Submit the production runs

```bash
bash cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b
bash cluster/bwunicluster3/submit_cv0_model.sh gpt_oss_20b
bash cluster/bwunicluster3/submit_cv0_model.sh llama31_8b
```

Each job requests one node, one task, one GPU and four CPU cores, with no
`--exclusive` and no explicit memory request. The helper submits the scientific
job exactly once with the profile's compatible partition list; Slurm chooses
placement. It does not use an idle-node probe. 27B is restricted to H100
production partitions; smoke-test its memory needs before production.

Overrides:

```bash
# explicit partition and walltime
SBATCH_GPU_PARTITION=gpu_h100_il \
SBATCH_GPU_TIME=32:00:00 \
bash cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b

# site account
SBATCH_ACCOUNT=your_project_account \
bash cluster/bwunicluster3/submit_cv0_model.sh gpt_oss_20b

# skip the prefetch job entirely
SKIP_MODEL_PREFETCH=1 \
bash cluster/bwunicluster3/submit_cv0_model.sh llama31_8b
```

### 4. Inspect the queue

```bash
squeue -u "$USER"
squeue -j JOB_ID
squeue --start -j JOB_ID       # projected start time
scancel JOB_ID
```

### 5. Follow a running job

```bash
tail -f cluster/bwunicluster3/logs/cv0-gemma4_26b_a4b-JOB_ID.out    # MIRAGE + job log
tail -f cluster/bwunicluster3/logs/cv0-gemma4_26b_a4b-JOB_ID.err    # errors only
tail -f ../vllm/logs/vllm-gemma4_26b_a4b-JOB_ID.log                 # vLLM server log
cat runs/_cluster_jobs/gemma4_26b_a4b-JOB_ID/cluster_job_metadata.json
cat runs/_cluster_jobs/gemma4_26b_a4b-JOB_ID/preflight.json
```

Expected connection refusals during model loading are suppressed; the job prints
a progress line about once a minute instead.

### 6. Results

```text
runs/cv0_gemma4_26b_a4b_100cp/<UTC-ts>_<config_hash8>/
runs/cv0_gpt_oss_20b_100cp/<UTC-ts>_<config_hash8>/
runs/cv0_llama31_8b_100cp/<UTC-ts>_<config_hash8>/
artifacts/<experiment_id>/
```

Each run directory also receives `vllm-server.log`, `cluster_job_metadata.json`
and `v1_models.json` from the job.

### 7. After the first measured run: tighten the walltimes

The walltimes for the three new models are first estimates. Once a run finishes:

```bash
sacct -j JOB_ID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Timelimit,MaxRSS,AllocTRES
```

Take `Elapsed`, add a margin (the Qwen reference run took ~16h52m against a 22h
request), and update `SBATCH_GPU_TIME_DEFAULT` in
`cluster/bwunicluster3/model_profiles/<profile>.sh`. Keep every default below 48
hours so the job stays eligible for both the standard and Ice Lake GPU queues.
