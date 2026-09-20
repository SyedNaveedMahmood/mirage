# CV-0 on bwUniCluster 3.0

This profile runs the full four-part CV-0 suite for **one model family per
allocation**, served by a separate vLLM environment on one GPU. The MIRAGE
client has its own environment, installed from this repository's
`pyproject.toml`. Model revisions, the AgentDojo revision and the vLLM version
are all pinned.

Six model arms across four families are available. They share every non-model scientific setting
(100 determinism checkpoints, K=16, four suites, two scaffolds, the same seeds,
horizons, thresholds and screens). This aligns the configured instrument, not
the sampled checkpoints or GPU hardware. See the
[code walkthrough](../../docs/CV0_IMPLEMENTATION.md) for implementation limits.

| Profile | Model | Revision | Parser | Default walltime | Gated |
| --- | --- | --- | --- | --- | --- |
| `qwen3p5_4b` | `Qwen/Qwen3.5-4B` | `851bf6e…8cd0a` | `qwen3_xml` | 22:00:00 | no |
| `qwen3p5_9b` | `Qwen/Qwen3.5-9B` | `c202236…37b9a` | `qwen3_xml` | 22:00:00 | no |
| `qwen3p5_27b` | `Qwen/Qwen3.5-27B` | `fc05dae…df654` | `qwen3_xml` | 47:00:00 | no |
| `gemma4_26b_a4b` | `google/gemma-4-26B-A4B-it` | `462a98a…5e6ae` | `gemma4` | 30:00:00 | **yes** |
| `gpt_oss_20b` | `openai/gpt-oss-20b` | `6cee5e8…febee` | `openai` | 30:00:00 | no |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f…c89659` | `llama3_json` | 22:00:00 | **yes** |

For sequential Qwen runs, use the [4B -> 9B -> 27B runbook](QWEN35_SEQUENCE.md):

```bash
bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --dry-run
# When ready (not part of the offline audit):
SBATCH_GPU_PARTITION=gpu_h100 bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh
```

The sequence attempts every size by default; `--stop-on-failure` instead
requires each predecessor to PASS. 27B is H100-only and needs a load/memory
smoke before production; its 47-hour allocation is an unmeasured budget.

## Layout of these scripts

```text
cluster/bwunicluster3/
├── cv0_model.sbatch            # THE shared production job body (all models)
├── prefetch_model.sbatch       # THE shared exact-revision prefetch job
├── submit_cv0_model.sh         # queue-aware production submission
├── submit_prefetch_model.sh    # prefetch only
├── submit_smoke_model.sh       # short development-queue smoke job
├── fetch_chat_template.sh      # pins the Llama 3.1 JSON tool-use template
├── model_profiles/<name>.sh    # per-model identity + vLLM flags (the only per-model file)
├── chat_templates/             # repository-controlled templates + provenance
├── lib/{common,profile,partition,port}.sh, lib/write_job_metadata.py
└── cv0_qwen3p5_9b.sbatch, prefetch_qwen3p5_9b.sbatch, submit_cv0.sh  # compatibility wrappers
```

Adding a model means adding one profile and two YAML configs — never another
copy of the Slurm script.

## Required cluster layout

Start in your existing `llmrun/` directory:

```bash
cd llmrun
mkdir -p agentrun
git clone --branch main \
  https://github.com/SyedNaveedMahmood/mirage.git \
  agentrun/mirage-persist
mkdir -p agentrun/vllm
cd agentrun/mirage-persist
```

The resulting layout is:

```text
llmrun/
└── agentrun/
    ├── vllm/                  # vLLM venv, caches, models, server logs
    └── mirage-persist/        # this checkout + independent MIRAGE venv
```

The scripts validate this topology and stop with a clear error if the checkout
is elsewhere. No computation or package build runs on a login node.

## 1. Create both environments (once)

```bash
cd llmrun/agentrun/mirage-persist
bash cluster/bwunicluster3/submit_setup.sh
squeue -u "$USER"
tail -f cluster/bwunicluster3/logs/mirage-setup-*.out
```

This produces `../vllm/.venv` with vLLM `0.25.1`, `./.venv` from
`pyproject.toml`, `.deps/agentdojo` at `089ed468cf3ed0322acc66b0211f26d9d90dbf60`,
and frozen package inventories for both environments.

Set `PYTHON_BIN=/path/to/python3.11` if `/usr/bin/python3.11` is not suitable.

## 2. Hugging Face access (gated models)

`google/gemma-4-26B-A4B-it` and `meta-llama/Llama-3.1-8B-Instruct` are gated.
Before prefetching them:

1. accept the licence on the model page with your Hugging Face account
   (`https://huggingface.co/google/gemma-4-26B-A4B-it`,
   `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`), and wait for
   approval if the repository requires manual review;
2. authenticate on a login node with the **same** account:

```bash
export HF_HOME="$PWD/../vllm/cache/huggingface"
../vllm/.venv/bin/hf auth login          # or: export HF_TOKEN=<token>
```

The prefetch job checks for credentials before downloading and explains exactly
what is missing. Tokens are never printed to any log.

## 3. Prefetch the model (CPU job, no GPU held)

```bash
./cluster/bwunicluster3/submit_prefetch_model.sh gemma4_26b_a4b
./cluster/bwunicluster3/submit_prefetch_model.sh gpt_oss_20b
./cluster/bwunicluster3/submit_prefetch_model.sh llama31_8b
```

Only the pinned revision is downloaded. On success a completion marker keyed on
model **and** revision is written to
`../vllm/cache/huggingface/.mirage_prefetch/<profile>@<revision>.done`; the job
is skipped entirely when that marker exists (`FORCE_MODEL_PREFETCH=1` overrides).
For `llama31_8b` the prefetch also pins the JSON tool-use chat template (see
`chat_templates/README.md`) — commit the three files it writes.

## 4. Short smoke test (development queue, optional)

```bash
./cluster/bwunicluster3/submit_smoke_model.sh llama31_8b
```

Loads the model, waits for `/health`, verifies `/v1/models`, runs one ordinary
generation and one tool call, then exits. It stays well inside the 30-minute
development limit, is named `smoke-<profile>`, and **never** runs the CV-0
campaign or chains another job. Production submissions never select a
development queue.

## 5. Submit the production run

```bash
./cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b
./cluster/bwunicluster3/submit_cv0_model.sh gpt_oss_20b
./cluster/bwunicluster3/submit_cv0_model.sh llama31_8b
```

Each submission:

* validates the single-model YAML against the selected profile's identity;
* offers one job to the profile's compatible production partitions, letting
  Slurm choose placement; it does not probe idle nodes to predict scheduling;
* requests exactly **1 node, 1 task, 1 GPU, 4 CPU cores**, with no `--exclusive`
  and no explicit `--mem`; the historical 9B memory observation does not
  establish 27B host-RAM sufficiency, which must be checked in a smoke;
* submits the scientific job **once**, with a compatible partition list;
* adds a CPU prefetch job with an `afterok` dependency only when the
  exact-revision marker is missing;
* prints the model, revision, partition, walltime and resource request before
  submitting, and the inspection commands afterwards.

### Useful overrides

```bash
# explicit partition and walltime
SBATCH_GPU_PARTITION=gpu_h100_il \
SBATCH_GPU_TIME=32:00:00 \
./cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b

# site account
SBATCH_ACCOUNT=my_project ./cluster/bwunicluster3/submit_cv0_model.sh gpt_oss_20b

# never submit a prefetch job (model is known to be cached)
SKIP_MODEL_PREFETCH=1 ./cluster/bwunicluster3/submit_cv0_model.sh llama31_8b

# Validate/preview any model without submitting (requires MIRAGE Python deps).
bash cluster/bwunicluster3/submit_cv0_model.sh llama31_8b --dry-run
```

Job-side overrides (`VLLM_PORT`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_MODEL_LEN`,
`VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_ENABLE_PREFIX_CACHING`,
`LLAMA31_CHAT_TEMPLATE`) are read inside the job and inherited from the
submitting shell.

`SBATCH_GPU_PARTITION` is validated: development queues and partitions outside
the compatible set are rejected for production runs.

`CV0_CONFIG=/absolute/path/to/config.yaml` is supported for a single-model
submission, but its model identity/revision must match the profile. It is
rejected by the Qwen sequence to prevent accidentally running one config three
times. `--check-only` checks real cluster prerequisites without submission;
`--parsable` prints only the accepted GPU job ID to stdout (diagnostics on stderr).
The sequence composes `--dependency afterany:ID` or `afterok:ID` with prefetch
success and logs accepted jobs using `--ledger`.

## 6. Monitor

```bash
squeue -u "$USER"                    # all your jobs
squeue -j JOB_ID                     # one job
squeue --start -j JOB_ID             # projected start time
scancel JOB_ID                       # cancel
sacct -j JOB_ID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Timelimit,MaxRSS,AllocTRES
```

## 7. Where everything lands

| What | Where |
| --- | --- |
| Slurm stdout | `cluster/bwunicluster3/logs/cv0-<profile>-<jobid>.out` |
| Slurm stderr | `cluster/bwunicluster3/logs/cv0-<profile>-<jobid>.err` |
| vLLM server log (live) | `../vllm/logs/vllm-<profile>-<jobid>.log` |
| Job artifacts | `runs/_cluster_jobs/<profile>-<jobid>/` |
| Experiment run dir | `runs/<experiment_id>/<UTC-ts>_<config_hash8>/` |
| Curated artifacts | `artifacts/<experiment_id>/` |

`runs/_cluster_jobs/<profile>-<jobid>/` holds the preserved server-side record:
`cluster_job_metadata.json` (Slurm job id, partition, requested walltime, CPU/GPU
counts, hostname, GPU identity, model id + revision, profile name, vLLM version,
tool parser, reasoning settings, chat-template path and SHA-256, selected port,
startup duration, `/v1/models` response, redacted launch command),
`vllm-server.log`, `nvidia-smi.txt`, `gpu_identity.csv`,
`scontrol_show_job.txt`, `vllm_launch_command.txt`, `mirage_command.txt`,
`preflight.json` and `cv0_exit_code.txt`. The server log, job metadata and
`/v1/models` response are also copied into the experiment's own run directory.
Credentials are never recorded.

## 8. Interpreting the exit code

`0 = PASS`, `1 = FAIL`, `2 = KILL`. A scientific FAIL/KILL therefore looks like a
failed Slurm job and must be interpreted with `cv0_verdict.json`, not diagnosed
as an infrastructure failure. Infrastructure failures (startup timeout, preflight
failure, gated access) print an explicit reason and preserve the server log.

## 9. Operational notes

* **Ports.** The job derives a deterministic port from the Slurm job id, probes
  it on the allocated node, advances if it is taken, and binds vLLM to
  `127.0.0.1`. Four one-GPU jobs can therefore share a four-GPU node. Set
  `VLLM_PORT` to force one.
* **Startup polling** is quiet: expected connection refusals are suppressed and a
  progress line is printed roughly once a minute. On timeout the tail of the
  server log is printed and the job exits non-zero.
* **Functional preflight.** Before the campaign starts, the job sends a few dozen
  tokens through the *same* backend MIRAGE uses: one ordinary completion and one
  request with a JSON tool schema whose response must parse as a tool call. A
  wrong parser, template or reasoning setting fails in seconds.
* **Walltimes** other than the historical 9B reference are planning estimates. After the first
  measured run, tighten `SBATCH_GPU_TIME_DEFAULT` in the model profile using the
  `Elapsed` value from `sacct` plus a margin. Keep every default below 48 hours
  so both the standard and Ice Lake queues remain eligible.

## Reduced validation runs

`configs/cv0/cv0_cluster_quick.yaml` is for short integration validation only. A
PASS from it is not a canonical or confirmatory CV-0 result.

## Compatibility with the Qwen-only entry points

`submit_cv0.sh`, `cv0_qwen3p5_9b.sbatch` and `prefetch_qwen3p5_9b.sbatch` still
work and now delegate to the shared implementation with the `qwen3p5_9b`
profile. The one behavioural change: `SBATCH_GPU_PARTITION` takes a single
partition name (the old comma-separated form is rejected with a pointer to
`submit_smoke_model.sh`).

## Cluster references

- [bwUniCluster 3.0 queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwUniCluster 3.0 GPU job syntax](https://wiki.bwhpc.de/e/BwUniCluster3.0/Slurm#GPU_jobs)
- [bwUniCluster 3.0 filesystems](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture)
- [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/)
