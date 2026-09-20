#!/usr/bin/env bash
#SBATCH --job-name=qwen35-fetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err
#
# COMPATIBILITY WRAPPER around the shared prefetch_model.sbatch. New work should
# use:  ./cluster/bwunicluster3/submit_prefetch_model.sh qwen3p5_9b

set -Eeuo pipefail

MIRAGE_SCRIPT_DIR="${MIRAGE_SCRIPT_DIR:-$(pwd -P)/cluster/bwunicluster3}"
export MIRAGE_SCRIPT_DIR
export MIRAGE_MODEL_PROFILE="qwen3p5_9b"

exec bash "${MIRAGE_SCRIPT_DIR}/prefetch_model.sbatch"
