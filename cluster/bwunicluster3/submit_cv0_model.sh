#!/usr/bin/env bash
# Submit the production CV-0 job for one model profile.
#
#   bash cluster/bwunicluster3/submit_cv0_model.sh gemma4_26b_a4b
#   Options: --dry-run, --check-only, --parsable, --dependency afterany:JOB_ID
#            (or afterok:JOB_ID), --ledger PATH
#
# Overrides (all optional):
#   SBATCH_GPU_PARTITION=gpu_h100_il   pin one compatible production partition
#   SBATCH_GPU_TIME=32:00:00           override the profile walltime
#   SBATCH_ACCOUNT=my_project          site account
#   SKIP_MODEL_PREFETCH=1              never submit the CPU prefetch job
#   CV0_CONFIG=/abs/path/config.yaml   override the scientific config
#
# The scientific job is submitted EXACTLY ONCE. By default it is offered to
# every production partition declared compatible by the model profile and Slurm
# chooses where it can start. SBATCH_GPU_PARTITION pins a single compatible
# partition when reproducibility or debugging requires that. Development queues
# are never selected here.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
usage() {
    printf 'usage: %s <model-profile> [--dry-run|--check-only] [--parsable] [--dependency afterany:ID|afterok:ID] [--ledger PATH]\n' "$0"
}
profile_name="${1:-}"
[[ "${profile_name}" != --help && "${profile_name}" != -h ]] || { usage; exit 0; }
[[ -n "${profile_name}" ]] || { usage >&2; exit 2; }
shift
dry_run=0
check_only=0
parsable=0
predecessor=""
ledger=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --check-only) check_only=1; shift ;;
        --parsable) parsable=1; shift ;;
        --dependency|--ledger)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            if [[ "$1" == --dependency ]]; then predecessor="$2"; else ledger="$2"; fi
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -n "${predecessor}" && ! "${predecessor}" =~ ^after(ok|any):[1-9][0-9]*$ ]]; then
    printf 'ERROR: dependency must be afterany:JOB_ID or afterok:JOB_ID\n' >&2
    exit 2
fi
if (( parsable )); then
    # A caller receives exactly one GPU job ID; diagnostics remain visible.
    exec 3>&1 1>&2
fi
if (( dry_run )); then
    # Preview anywhere, including a workstation outside the cluster layout.
    # No Slurm, environment setup, files, downloads or server requests.
    MIRAGE_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
    VLLM_ROOT="${VLLM_ROOT:-$(dirname -- "${MIRAGE_REPO_ROOT}")/vllm}"
    die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
else
    # shellcheck source=lib/common.sh
    source "${SCRIPT_DIR}/lib/common.sh"
    require_command sbatch
    [[ -x "${VLLM_ROOT}/.venv/bin/vllm" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"
    [[ -x "${MIRAGE_REPO_ROOT}/.venv/bin/mirage" ]] || die "run ${SCRIPT_DIR}/submit_setup.sh and wait for it first"
fi
# shellcheck source=lib/profile.sh
source "${SCRIPT_DIR}/lib/profile.sh"
# shellcheck source=lib/partition.sh
source "${SCRIPT_DIR}/lib/partition.sh"

load_model_profile "${profile_name}" || exit 2

cv0_config="${CV0_CONFIG:-${MIRAGE_REPO_ROOT}/${CV0_CONFIG_DEFAULT}}"
[[ -f "${cv0_config}" ]] || die "CV-0 config not found: ${cv0_config}"
# Slurm --export uses commas as delimiters; refuse ambiguous values.
for value in "${SCRIPT_DIR}" "${cv0_config}"; do
    [[ "${value}" != *','* && "${value}" != *$'\n'* ]] || die "paths must not contain commas or newlines"
done
config_python="${MIRAGE_REPO_ROOT}/.venv/bin/python"
if (( dry_run )); then
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        config_python="${PYTHON_BIN}"
    elif [[ -x "${MIRAGE_REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
        config_python="${MIRAGE_REPO_ROOT}/.venv/Scripts/python.exe"
    elif [[ ! -x "${config_python}" ]]; then
        config_python="$(command -v python3 || command -v python)" || die "set PYTHON_BIN to a MIRAGE Python environment"
    fi
fi
PYTHONDONTWRITEBYTECODE=1 "${config_python}" "${SCRIPT_DIR}/lib/validate_profile_config.py" \
    "${cv0_config}" "${MODEL_ID}" "${MODEL_REVISION}" "${SERVED_MODEL_NAME}" "${MODEL_FAMILY}"

# --- partition --------------------------------------------------------------
candidates="${PROFILE_PARTITION_PREFERENCE}"
if [[ -n "${SBATCH_GPU_PARTITION:-}" ]]; then
    partition="${SBATCH_GPU_PARTITION}"
    partition_reason="explicitly pinned via SBATCH_GPU_PARTITION"
    validate_production_partition "${partition}" "${candidates}" \
        || die "refusing to submit the production run to ${partition}"
else
    # Do not infer future availability from a snapshot of currently idle nodes.
    # Submit once against every reviewed, model-compatible production partition
    # and let Slurm choose the placement/start time.
    partition="$(slurm_partition_list "${candidates}")" \
        || die "no compatible production partitions configured for ${PROFILE_NAME}"
    partition_reason="all profile-compatible production partitions; Slurm selects placement"
fi

walltime="${SBATCH_GPU_TIME:-${SBATCH_GPU_TIME_DEFAULT}}"
[[ "${walltime}" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] \
    || die "SBATCH_GPU_TIME must use HH:MM:SS"
wall_seconds=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
(( wall_seconds > 0 && wall_seconds < 48 * 3600 )) || die "production walltime must be positive and below 48h"

# --- prefetch dependency ----------------------------------------------------
hf_home="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
marker_file="${hf_home}/.mirage_prefetch/${PROFILE_NAME}@${MODEL_REVISION}.done"

common_args=(--chdir="${MIRAGE_REPO_ROOT}")
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    common_args+=(--account="${SBATCH_ACCOUNT}")
fi
export_common="ALL,MIRAGE_MODEL_PROFILE=${PROFILE_NAME},MIRAGE_SCRIPT_DIR=${SCRIPT_DIR}"

dependency="${predecessor}"
prefetch_id=""
prefetch_needed=1
if [[ "${SKIP_MODEL_PREFETCH:-0}" == 1 ]]; then
    prefetch_needed=0
    printf 'Prefetch: skipped (SKIP_MODEL_PREFETCH=1)\n'
elif [[ -f "${marker_file}" ]]; then
    prefetch_needed=0
    printf 'Prefetch: already complete for this exact revision (%s); no CPU job submitted\n' \
        "${marker_file}"
fi

printf 'Plan: %s @ %s; config=%s; partitions=%s; time=%s; predecessor=%s\n' \
    "${MODEL_ID}" "${MODEL_REVISION}" "${cv0_config}" "${partition}" "${walltime}" "${predecessor:-none}"
if (( dry_run || check_only )); then
    printf 'Validation only: CPU prefetch needed=%s; no jobs submitted.\n' "${prefetch_needed}"
    exit 0
fi

# Slurm opens logs at job start; these directories must exist before submission.
mkdir -p "${SCRIPT_DIR}/logs" "${VLLM_ROOT}/logs" "${MIRAGE_REPO_ROOT}/runs/_cluster_jobs"
if [[ -n "${ledger}" ]]; then
    [[ -f "${ledger}" && -w "${ledger}" ]] || die "ledger must be an existing writable file: ${ledger}"
fi
record_job() {
    if [[ -n "${ledger}" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "${PROFILE_NAME}" "$1" "$2" "$3" "${cv0_config}" >> "${ledger}"
    fi
}
validate_job_id() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID: $1; inspect squeue before retrying"
}
if (( prefetch_needed )); then
    prefetch_id="$(sbatch --parsable "${common_args[@]}" \
        --job-name="fetch-${PROFILE_NAME}" \
        --partition="${SBATCH_CPU_PARTITION:-cpu}" \
        --export="${export_common}" \
        "${SCRIPT_DIR}/prefetch_model.sbatch")"
    prefetch_id="${prefetch_id%%;*}"
    validate_job_id "${prefetch_id}"
    record_job prefetch "${prefetch_id}" none
    dependency="${dependency:+${dependency},}afterok:${prefetch_id}"
    printf 'Submitted model prefetch job: %s\n' "${prefetch_id}"
fi
dependency_args=()
if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="${dependency}" --kill-on-invalid-dep=yes)
fi

# --- summary ----------------------------------------------------------------
printf '\n=== CV-0 production submission ==========================================\n'
printf 'model profile   : %s (%s)\n' "${PROFILE_NAME}" "${PROFILE_DESCRIPTION:-}"
printf 'model           : %s\n' "${MODEL_ID}"
printf 'revision        : %s\n' "${MODEL_REVISION}"
printf 'CV-0 config     : %s\n' "${cv0_config}"
printf 'partition(s)    : %s (%s)\n' "${partition}" "${partition_reason}"
printf 'walltime        : %s\n' "${walltime}"
printf 'resources       : 1 node, 1 task, 1 GPU, 4 CPU cores, no --exclusive, no explicit --mem\n'
printf 'tool parser     : %s\n' "${VLLM_TOOL_CALL_PARSER}"
printf 'prefetch job    : %s\n' "${prefetch_id:-none}"
printf '=========================================================================\n\n'

if ! gpu_id="$(sbatch --parsable "${common_args[@]}" "${dependency_args[@]}" \
    --job-name="cv0-${PROFILE_NAME}" \
    --partition="${partition}" \
    --time="${walltime}" \
    --export="${export_common},MIRAGE_JOB_MODE=production,CV0_CONFIG=${cv0_config}" \
    "${SCRIPT_DIR}/cv0_model.sbatch")"; then
    printf 'GPU submission failed. Already accepted prefetch job: %s (not cancelled).\n' "${prefetch_id:-none}" >&2
    exit 1
fi
gpu_id="${gpu_id%%;*}"
validate_job_id "${gpu_id}"
record_job gpu "${gpu_id}" "${dependency:-none}"

printf 'Submitted CV-0 GPU job: %s\n' "${gpu_id}"
printf 'Queue status     : squeue -j %s\n' "${gpu_id}"
printf 'Projected start  : squeue --start -j %s\n' "${gpu_id}"
printf 'Cancel           : scancel %s\n' "${gpu_id}"
printf 'Slurm stdout     : %s/cv0-%s-%s.out\n' "${SCRIPT_DIR}/logs" "${PROFILE_NAME}" "${gpu_id}"
printf 'Slurm stderr     : %s/cv0-%s-%s.err\n' "${SCRIPT_DIR}/logs" "${PROFILE_NAME}" "${gpu_id}"
printf 'Job artifacts    : %s/runs/_cluster_jobs/%s-%s/\n' "${MIRAGE_REPO_ROOT}" "${PROFILE_NAME}" "${gpu_id}"
if (( parsable )); then
    printf '%s\n' "${gpu_id}" >&3
fi
