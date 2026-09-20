#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
require_command sbatch

mkdir -p "${SCRIPT_DIR}/logs"
args=(--chdir="${MIRAGE_REPO_ROOT}" --partition="${SBATCH_CPU_PARTITION:-cpu}")
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    args+=(--account="${SBATCH_ACCOUNT}")
fi

job_id="$(sbatch --parsable "${args[@]}" "${SCRIPT_DIR}/setup_environments.sbatch")"
printf 'Submitted environment setup job: %s\n' "${job_id%%;*}"
