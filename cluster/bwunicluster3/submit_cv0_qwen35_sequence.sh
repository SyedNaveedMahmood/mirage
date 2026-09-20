#!/usr/bin/env bash
# Submit separate allocations in order: 4B -> 9B -> 27B. No login-node inference.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dry_run=0
dependency_kind=afterany
usage() {
    printf 'usage: bash %s [--dry-run] [--stop-on-failure]\n' "$0"
    printf 'Default: attempt all three sizes sequentially, even after FAIL/KILL or infrastructure failure.\n'
    printf '%s\n' '--stop-on-failure: require each preceding GPU job to exit zero (CV-0 PASS).'
}
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --stop-on-failure) dependency_kind=afterok ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done
if [[ -n "${CV0_CONFIG:-}" ]]; then
    printf 'ERROR: unset CV0_CONFIG; the sequence uses a distinct pinned config for each size.\n' >&2
    exit 2
fi
profiles=(qwen3p5_4b qwen3p5_9b qwen3p5_27b)
submit="${SCRIPT_DIR}/submit_cv0_model.sh"
check_flag=--check-only
if (( dry_run )); then check_flag=--dry-run; fi
printf 'Qwen3.5 sequence: 4B -> 9B -> 27B; GPU dependency policy=%s\n' "${dependency_kind}"
# Prevalidate ALL three before the first sbatch. Each child gets a clean profile
# scope; hooks/variables from one model cannot leak to another.
for profile in "${profiles[@]}"; do
    bash "${submit}" "${profile}" "${check_flag}"
done
if (( dry_run )); then
    printf 'GPU chain: 4B --%s--> 9B --%s--> 27B; each also requires its own prefetch success.\n' \
        "${dependency_kind}" "${dependency_kind}"
    printf 'Dry-run complete: no jobs, servers, downloads or output directories created.\n'
    exit 0
fi

repo_root="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
mkdir -p "${repo_root}/runs/_cluster_campaigns"
ledger="$(mktemp "${repo_root}/runs/_cluster_campaigns/qwen35-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.tsv")"
printf 'profile\tkind\tjob_id\tdependency\tconfig\n' > "${ledger}"
printf 'Accepted-job ledger: %s\n' "${ledger}"
submission_complete=0
on_exit() {
    local status=$?
    if (( ! submission_complete )); then
        printf 'Sequence submission incomplete (exit %s). Accepted jobs are NOT rolled back.\n' "${status}" >&2
        printf 'Inspect %s and squeue before retrying; cancel unwanted job IDs with scancel.\n' "${ledger}" >&2
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

previous=""
gpu_ids=()
for profile in "${profiles[@]}"; do
    args=("${profile}" --parsable --ledger "${ledger}")
    if [[ -n "${previous}" ]]; then
        args+=(--dependency "${dependency_kind}:${previous}")
    fi
    previous="$(bash "${submit}" "${args[@]}")"
    [[ "${previous}" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: invalid GPU job ID\n' >&2; exit 1; }
    gpu_ids+=("${previous}")
    printf 'Queued %s as GPU job %s\n' "${profile}" "${previous}"
done
submission_complete=1
printf 'All three submitted, not yet completed. Monitor: squeue -j %s\n' "$(IFS=,; echo "${gpu_ids[*]}")"
printf 'Final states: sacct -j %s --format=JobID,JobName,State,ExitCode,Elapsed\n' "$(IFS=,; echo "${gpu_ids[*]}")"
printf 'No aggregate scientific verdict is inferred; inspect each run and %s.\n' "${ledger}"
