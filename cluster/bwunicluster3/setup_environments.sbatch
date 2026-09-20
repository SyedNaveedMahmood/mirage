#!/usr/bin/env bash
#SBATCH --job-name=mirage-setup
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="$(pwd -P)"
cd "${REPO_ROOT}"
"${REPO_ROOT}/cluster/bwunicluster3/setup_all.sh"
