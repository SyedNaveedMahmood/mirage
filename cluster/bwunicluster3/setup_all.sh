#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

"${SCRIPT_DIR}/setup_vllm_env.sh"
"${SCRIPT_DIR}/setup_mirage_env.sh"

printf '\nBoth isolated environments are ready.\n'
