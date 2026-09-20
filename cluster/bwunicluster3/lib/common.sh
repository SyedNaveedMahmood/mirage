#!/usr/bin/env bash

# Shared path validation for the bwUniCluster 3.0 scripts. The intended layout is:
#   <llmrun>/agentrun/vllm
#   <llmrun>/agentrun/mirage-persist   (this repository)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

lowercase() {
    tr '[:upper:]' '[:lower:]'
}

discover_layout() {
    local common_dir detected_repo expected_repo
    common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    detected_repo="$(git -C "${common_dir}" rev-parse --show-toplevel 2>/dev/null)" \
        || die "these scripts must be run from a Git checkout"

    export MIRAGE_REPO_ROOT="${MIRAGE_REPO_ROOT:-${detected_repo}}"
    export AGENTRUN_ROOT="${AGENTRUN_ROOT:-$(dirname -- "${MIRAGE_REPO_ROOT}")}"
    export VLLM_ROOT="${VLLM_ROOT:-${AGENTRUN_ROOT}/vllm}"
    expected_repo="${AGENTRUN_ROOT}/mirage-persist"

    [[ "$(basename -- "${AGENTRUN_ROOT}" | lowercase)" == "agentrun" ]] \
        || die "repository must be under <llmrun>/agentrun (found ${AGENTRUN_ROOT})"
    [[ "$(basename -- "${MIRAGE_REPO_ROOT}" | lowercase)" == "mirage-persist" ]] \
        || die "clone this repository as ${expected_repo}"
    [[ "$(cd -- "${MIRAGE_REPO_ROOT}" && pwd -P)" == "$(cd -- "${detected_repo}" && pwd -P)" ]] \
        || die "MIRAGE_REPO_ROOT does not identify this checkout"
}

select_python() {
    local candidate
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        [[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
        printf '%s\n' "${PYTHON_BIN}"
        return
    fi
    for candidate in /usr/bin/python3.11 python3.11 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            command -v "${candidate}"
            return
        fi
    done
    die "Python >=3.11 was not found; set PYTHON_BIN to a suitable interpreter"
}

discover_layout
