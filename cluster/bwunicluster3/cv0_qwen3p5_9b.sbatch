#!/usr/bin/env bash
#SBATCH --job-name=cv0-qwen35-9b
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err
#
# COMPATIBILITY WRAPPER. The Qwen job body now lives in the shared
# cv0_model.sbatch, driven by model_profiles/qwen3p5_9b.sh. Existing commands
# that submit this file keep working; new work should use:
#
#   ./cluster/bwunicluster3/submit_cv0_model.sh qwen3p5_9b
#
# Resource note: this wrapper follows the current policy (4 CPU cores, no
# explicit --mem, no --exclusive). The reference run peaked at ~7.2 GB RSS with
# ~1.5% CPU efficiency, so the previous 16-core/128 GB request only delayed
# scheduling.

set -Eeuo pipefail

MIRAGE_SCRIPT_DIR="${MIRAGE_SCRIPT_DIR:-$(pwd -P)/cluster/bwunicluster3}"
export MIRAGE_SCRIPT_DIR
export MIRAGE_MODEL_PROFILE="qwen3p5_9b"
export MIRAGE_JOB_MODE="${MIRAGE_JOB_MODE:-production}"

exec bash "${MIRAGE_SCRIPT_DIR}/cv0_model.sbatch"
