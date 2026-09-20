# Qwen3.5 4B -> 9B -> 27B CV-0 sequence

This submits three separate 100-checkpoint, K=16 CV-0 jobs through the shared
cluster runner. It does not combine the three sizes into independent families.
Read the [code walkthrough](../../docs/CV0_IMPLEMENTATION.md) and the resource
limits below before interpreting results. 4B/27B still require GPU validation.

## Setup and preview

Use the [standard setup](README.md) under
`llmrun/agentrun/mirage-persist`, cloned from
`https://github.com/SyedNaveedMahmood/mirage.git` on `main`. Both environments
must be ready before real submission. A dry-run needs the MIRAGE Python/config
dependencies but no Slurm, vLLM installation, model download or cluster layout:

```bash
bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --dry-run
```

On Windows, run that command in Git Bash; the script detects
`.venv/Scripts/python.exe`. Alternatively set `PYTHON_BIN` to your installed
MIRAGE environment's Python. Preview resolves/validates every config and checks
its model ID/revision/family against the profile. It does not test GPU memory,
partition availability, Hugging Face access, or site scheduling policy.

## Short cluster validation (not a full experiment)

For each profile, prefetch first, wait for success, then run the existing
development-queue smoke. Example for the largest model:

```bash
bash cluster/bwunicluster3/submit_prefetch_model.sh qwen3p5_27b
# Wait for the printed CPU job to complete successfully.
bash cluster/bwunicluster3/submit_smoke_model.sh qwen3p5_27b
```

Repeat for `qwen3p5_4b` and `qwen3p5_9b`. Smoke submits only one model; it loads
vLLM, checks health/model identity and makes plain/tool-call API probes. It never
starts CV-0. Default `dev_gpu_h100` is appropriate; do not select a 40-GiB GPU
for 27B. The 30-minute smoke allocation includes all startup and probes, so it
can time out even though the profile's polling budget is also 30 minutes.

Inspect `runs/_cluster_jobs/<profile>-<jobid>/preflight.json`,
`gpu_identity.csv`, `vllm-server.log` and `sacct` memory/state information.
Host memory follows the site's default for the existing four-CPU allocation;
27B loading/RSS has not been measured. Failed smoke is a reason to investigate
before production, not to lower K or the scientific gate thresholds.

## Submit the sequence when ready

```bash
# Recommended for comparison: hold GPU architecture/partition constant.
SBATCH_GPU_PARTITION=gpu_h100 \
  bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh
```

The script validates all three profiles/configs/environments/partitions before
its first submission. It then submits the following dependency graph (CPU
prefetches are omitted when an exact-revision completion marker exists):

```text
fetch 4B --afterok--> GPU 4B --afterany--> GPU 9B --afterany--> GPU 27B
                                        ^                   ^
fetch 9B ----------------afterok---------+                   |
fetch 27B --------------------------------afterok-----------+
```

GPU allocations do not overlap within this sequence; CPU downloads can overlap.
Each GPU allocation starts its own server, runs preflight and CV-0, then cleans
up that server before Slurm releases the dependency. Scientific failures are
data: the default `afterany` policy attempts later sizes after PASS, FAIL, KILL,
cancellation or infrastructure failure. It does not retry a failed model.
This policy does **not** waive CV-0 gates or authorize later experiments.

To stop later sizes when any preceding GPU job exits nonzero:

```bash
SBATCH_GPU_PARTITION=gpu_h100 \
  bash cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh --stop-on-failure
```

This uses `afterok` between GPUs: CV-0 FAIL/KILL and infrastructure failures all
block successors. Prefetch always requires `afterok`. Jobs whose dependencies
become impossible use `--kill-on-invalid-dep=yes` so Slurm cancels them rather
than leaving them pending indefinitely. See the official
[sbatch dependency documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

Submission is not transactional. If a later `sbatch` call fails, the script
exits immediately, prints the ledger path, and leaves already accepted jobs
alone. Inspect them before retrying; repeating the command creates a new
campaign, not a resume. Invalid job-ID responses require checking `squeue`
because Slurm may have accepted a job whose response could not be recorded.

## Resources, overrides and results

| Profile | Default GPU partitions | Walltime | GPU utilization | Startup budget |
| --- | --- | --- | --- | --- |
| `qwen3p5_4b` | `gpu_h100,gpu_h100_il,gpu_a100_il` | 22h | 0.85 | 1200s |
| `qwen3p5_9b` | `gpu_h100,gpu_h100_il,gpu_a100_il` | 22h | 0.85 | 1200s |
| `qwen3p5_27b` | `gpu_h100,gpu_h100_il` | 47h | 0.90 | 1800s |

All request one GPU/four CPUs; context 32768, TP=1, max sequences 1, BF16,
non-thinking, text-only. 27B resource sufficiency and runtime remain unmeasured.
The current shared client is synchronous; adding concurrency is not part of
this size extension.

Inherited options: `SBATCH_ACCOUNT`, `SBATCH_CPU_PARTITION`, `HF_HOME`,
`SKIP_MODEL_PREFETCH=1`, `SBATCH_GPU_PARTITION`, `SBATCH_GPU_TIME`, and existing
vLLM environment overrides. GPU partition overrides must work for **every**
size; the whole sequence rejects `gpu_a100_il` because 27B is H100-only.
Walltime overrides use `HH:MM:SS`, must be positive and below 48 hours, and
apply to all three sizes. Leave them unset to retain size-specific defaults.
Use serving overrides deliberately and retain the resulting launch metadata.
`CV0_CONFIG` must be unset: each size has its own pinned configuration.

The wrapper returns after submission, not after completion. Its TSV ledger is
`runs/_cluster_campaigns/qwen35-<UTC-time>-<unique>.tsv`, with one row for each
accepted prefetch/GPU job and its dependencies/config path. It prints exact
`squeue` and `sacct` commands. Cancel **all relevant IDs from the ledger** to
stop a campaign; cancelling just 4B does not stop default-afterany successors.

Results are independent:

```text
runs/cv0_qwen3p5_4b_100cp/<timestamp>_<config-hash>/
runs/cv0_qwen3p5_9b_100cp/<timestamp>_<config-hash>/
runs/cv0_qwen3p5_27b_100cp/<timestamp>_<config-hash>/
runs/_cluster_jobs/<profile>-<jobid>/
```

Read each `cv0_verdict.json` and the Slurm terminal state; absence of a verdict
is not a PASS. The sequence does not merge results or manufacture a combined
scientific verdict. Keep the checkout unchanged until all jobs finish.
