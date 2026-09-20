#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_command git

VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
VLLM_ENV="${VLLM_ROOT}/.venv"
PYTHON="$(select_python)"

mkdir -p "${VLLM_ROOT}" "${VLLM_ROOT}/cache/pip" "${VLLM_ROOT}/cache/huggingface" "${VLLM_ROOT}/logs"
export PIP_CACHE_DIR="${VLLM_ROOT}/cache/pip"

if [[ ! -x "${VLLM_ENV}/bin/python" ]]; then
    "${PYTHON}" -m venv "${VLLM_ENV}"
fi

"${VLLM_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
# Refuse an accidental source build on a login/CPU node; vLLM publishes Linux wheels.
"${VLLM_ENV}/bin/python" -m pip install --upgrade --only-binary=vllm \
    "vllm==${VLLM_VERSION}" "huggingface_hub>=0.36"
"${VLLM_ENV}/bin/python" -m pip check
"${VLLM_ENV}/bin/python" -m pip freeze > "${VLLM_ROOT}/requirements.lock.txt"

installed_version="$("${VLLM_ENV}/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
[[ "${installed_version}" == "${VLLM_VERSION}" ]] \
    || die "installed vLLM ${installed_version}, expected ${VLLM_VERSION}"

printf 'vLLM environment ready: %s (vLLM %s)\n' "${VLLM_ENV}" "${installed_version}"
