#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_command git

AGENTDOJO_URL="${AGENTDOJO_URL:-https://github.com/ethz-spylab/agentdojo.git}"
AGENTDOJO_SHA="${AGENTDOJO_SHA:-089ed468cf3ed0322acc66b0211f26d9d90dbf60}"
AGENTDOJO_DIR="${MIRAGE_REPO_ROOT}/.deps/agentdojo"
MIRAGE_ENV="${MIRAGE_REPO_ROOT}/.venv"
PYTHON="$(select_python)"

mkdir -p "${MIRAGE_REPO_ROOT}/.deps" "${VLLM_ROOT}/cache/pip"
export PIP_CACHE_DIR="${VLLM_ROOT}/cache/pip"

if [[ ! -d "${AGENTDOJO_DIR}/.git" ]]; then
    [[ ! -e "${AGENTDOJO_DIR}" ]] \
        || die "${AGENTDOJO_DIR} exists but is not a Git checkout"
    git clone --filter=blob:none "${AGENTDOJO_URL}" "${AGENTDOJO_DIR}"
fi
git -C "${AGENTDOJO_DIR}" fetch --depth=1 origin "${AGENTDOJO_SHA}"
git -C "${AGENTDOJO_DIR}" switch --detach "${AGENTDOJO_SHA}"

if [[ ! -x "${MIRAGE_ENV}/bin/python" ]]; then
    "${PYTHON}" -m venv "${MIRAGE_ENV}"
fi

"${MIRAGE_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${MIRAGE_ENV}/bin/python" -m pip install --editable "${AGENTDOJO_DIR}"

project_spec="${MIRAGE_REPO_ROOT}"
if [[ -n "${MIRAGE_EXTRAS:-}" ]]; then
    project_spec="${MIRAGE_REPO_ROOT}[${MIRAGE_EXTRAS}]"
fi
# The MIRAGE environment is resolved from the repository's pyproject.toml.
"${MIRAGE_ENV}/bin/python" -m pip install --editable "${project_spec}"
"${MIRAGE_ENV}/bin/python" -m pip check
"${MIRAGE_ENV}/bin/python" -m pip freeze > "${MIRAGE_REPO_ROOT}/cluster_env_freeze.txt"

installed_sha="$(git -C "${AGENTDOJO_DIR}" rev-parse HEAD)"
[[ "${installed_sha}" == "${AGENTDOJO_SHA}" ]] \
    || die "AgentDojo checkout ${installed_sha}, expected ${AGENTDOJO_SHA}"
"${MIRAGE_ENV}/bin/mirage" --help >/dev/null

printf 'MIRAGE environment ready: %s\n' "${MIRAGE_ENV}"
printf 'AgentDojo revision: %s\n' "${installed_sha}"
