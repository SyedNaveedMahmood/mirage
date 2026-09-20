#!/usr/bin/env bash
# Submit ONLY the CPU model-prefetch job for one model profile.
#
#   ./cluster/bwunicluster3/submit_prefetch_model.sh gemma4_26b_a4b
#
# Useful before a development-queue smoke test, or to get a gated-access failure
# out of the way without holding a GPU. The job is skipped entirely when the
# exact-revision completion marker already exists (FORCE_MODEL_PREFETCH=1
# re-downloads anyway).
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

mkdir -p "${SCRIPT_DIR}/logs"

hf_home="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
marker_file="${hf_home}/.mirage_prefetch/${PROFILE_NAME}@${MODEL_REVISION}.done"
if [[ -f "${marker_file}" && "${FORCE_MODEL_PREFETCH:-0}" != 1 ]]; then
    printf 'Already prefetched at the pinned revision; nothing submitted.\n'
    printf 'Marker: %s\n' "${marker_file}"
    exit 0
fi

common_args=(--chdir="${MIRAGE_REPO_ROOT}")
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    common_args+=(--account="${SBATCH_ACCOUNT}")
fi

job_id="$(sbatch --parsable "${common_args[@]}" \
    --job-name="fetch-${PROFILE_NAME}" \
    --partition="${SBATCH_CPU_PARTITION:-cpu}" \
    --export="ALL,MIRAGE_MODEL_PROFILE=${PROFILE_NAME},MIRAGE_SCRIPT_DIR=${SCRIPT_DIR},FORCE_MODEL_PREFETCH=${FORCE_MODEL_PREFETCH:-0}" \
    "${SCRIPT_DIR}/prefetch_model.sbatch")"
job_id="${job_id%%;*}"

printf 'Submitted prefetch job: %s (%s@%s)\n' "${job_id}" "${MODEL_ID}" "${MODEL_REVISION}"
printf 'Queue status     : squeue -j %s\n' "${job_id}"
printf 'Projected start  : squeue --start -j %s\n' "${job_id}"
printf 'Slurm stdout     : %s/fetch-%s-%s.out\n' "${SCRIPT_DIR}/logs" "${PROFILE_NAME}" "${job_id}"
