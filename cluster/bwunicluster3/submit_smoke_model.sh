#!/usr/bin/env bash
# Submit a SHORT development-queue smoke job for one model profile.
#
#   ./cluster/bwunicluster3/submit_smoke_model.sh llama31_8b
#
# It verifies model loading, server health, one ordinary generation and one tool
# call, then exits. It NEVER runs the CV-0 campaign, never chains a follow-up
# job, and is named `smoke-<profile>` so it cannot be mistaken for a production
# run. Production submissions (submit_cv0_model.sh) never select a development
# queue.
#
# The development queue allows one running job per user with a 30-minute limit,
# so the model must already be prefetched: download time would consume the whole
# allocation.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/profile.sh
source "${SCRIPT_DIR}/lib/profile.sh"

require_command sbatch

profile_name="${1:-}"
if [[ -z "${profile_name}" ]]; then
    printf 'usage: %s <model-profile>\n' "$0" >&2
    printf 'available profiles: %s\n' "$(list_model_profiles)" >&2
    exit 2
fi
load_model_profile "${profile_name}" || exit 2

[[ -x "${VLLM_ROOT}/.venv/bin/vllm" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"
[[ -x "${MIRAGE_REPO_ROOT}/.venv/bin/mirage" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"

cv0_config="${CV0_CONFIG:-${MIRAGE_REPO_ROOT}/${CV0_CONFIG_DEFAULT}}"
[[ -f "${cv0_config}" ]] || die "CV-0 config not found: ${cv0_config}"

hf_home="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
marker_file="${hf_home}/.mirage_prefetch/${PROFILE_NAME}@${MODEL_REVISION}.done"
if [[ ! -f "${marker_file}" && "${SKIP_MODEL_PREFETCH:-0}" != 1 ]]; then
    printf 'ERROR: %s@%s is not prefetched (%s missing).\n' \
        "${MODEL_ID}" "${MODEL_REVISION}" "${marker_file}" >&2
    printf 'Run the prefetch job first:\n' >&2
    printf '  bash %s/submit_prefetch_model.sh %s\n' "${SCRIPT_DIR}" "${PROFILE_NAME}" >&2
    printf 'or set SKIP_MODEL_PREFETCH=1 to accept downloading inside the 30-minute limit.\n' >&2
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs" "${VLLM_ROOT}/logs" "${MIRAGE_REPO_ROOT}/runs/_cluster_jobs"

# Development GPU partitions only; 30-minute site limit.
partition="${SBATCH_SMOKE_PARTITION:-dev_gpu_h100}"
walltime="${SBATCH_SMOKE_TIME:-00:30:00}"
if [[ ! "${partition}" =~ ^dev_ ]]; then
    die "submit_smoke_model.sh is for development queues only, got '${partition}'"
fi

common_args=(--chdir="${MIRAGE_REPO_ROOT}")
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    common_args+=(--account="${SBATCH_ACCOUNT}")
fi

printf '\n=== CV-0 SMOKE submission (development queue, no campaign) ==============\n'
printf 'model profile   : %s (%s)\n' "${PROFILE_NAME}" "${PROFILE_DESCRIPTION:-}"
printf 'model           : %s@%s\n' "${MODEL_ID}" "${MODEL_REVISION}"
printf 'partition       : %s\n' "${partition}"
printf 'walltime        : %s\n' "${walltime}"
printf 'resources       : 1 node, 1 task, 1 GPU, 4 CPU cores\n'
printf 'scope           : model load + /health + /v1/models + generation + 1 tool call\n'
printf '=========================================================================\n\n'

job_id="$(sbatch --parsable "${common_args[@]}" \
    --job-name="smoke-${PROFILE_NAME}" \
    --partition="${partition}" \
    --time="${walltime}" \
    --export="ALL,MIRAGE_MODEL_PROFILE=${PROFILE_NAME},MIRAGE_SCRIPT_DIR=${SCRIPT_DIR},MIRAGE_JOB_MODE=smoke,CV0_CONFIG=${cv0_config}" \
    "${SCRIPT_DIR}/cv0_model.sbatch")"
job_id="${job_id%%;*}"

printf 'Submitted SMOKE job: %s\n' "${job_id}"
printf 'Queue status     : squeue -j %s\n' "${job_id}"
printf 'Projected start  : squeue --start -j %s\n' "${job_id}"
printf 'Cancel           : scancel %s\n' "${job_id}"
printf 'Slurm stdout     : %s/smoke-%s-%s.out\n' "${SCRIPT_DIR}/logs" "${PROFILE_NAME}" "${job_id}"
