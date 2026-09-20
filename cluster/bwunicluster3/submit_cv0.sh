#!/usr/bin/env bash
# COMPATIBILITY WRAPPER: the original Qwen-only entry point.
#
#   bash cluster/bwunicluster3/submit_cv0.sh
#
# is now equivalent to
#
#   bash cluster/bwunicluster3/submit_cv0_model.sh qwen3p5_9b
#
# All previously supported environment overrides (SBATCH_ACCOUNT,
# SBATCH_GPU_PARTITION, SBATCH_GPU_TIME, SBATCH_CPU_PARTITION,
# SKIP_MODEL_PREFETCH, CV0_CONFIG) still apply.
#
# Note: SBATCH_GPU_PARTITION is now a single partition name. The old
# comma-separated "try several queues" form is rejected, because the reduced
# development-queue workflow it was used for has its own entry point:
#   bash cluster/bwunicluster3/submit_smoke_model.sh qwen3p5_9b
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "${SBATCH_GPU_PARTITION:-}" == *,* ]]; then
    printf 'ERROR: SBATCH_GPU_PARTITION now takes a single partition name, got: %s\n' \
        "${SBATCH_GPU_PARTITION}" >&2
    printf 'For a short development-queue run use:\n' >&2
    printf '  bash %s/submit_smoke_model.sh qwen3p5_9b\n' "${SCRIPT_DIR}" >&2
    exit 2
fi

exec bash "${SCRIPT_DIR}/submit_cv0_model.sh" "${MIRAGE_MODEL_PROFILE:-qwen3p5_9b}"
