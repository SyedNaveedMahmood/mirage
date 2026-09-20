#!/usr/bin/env bash

# Model-profile loading for the shared CV-0 cluster jobs.
#
# A profile is a small, reviewed shell fragment under model_profiles/ that
# declares one model's identity and its vLLM launch specifics. Everything else
# (resources, port selection, health polling, preflight, provenance, the CV-0
# invocation) lives once in cv0_model.sbatch. Adding a model must not mean
# copying a Slurm script.
#
# Profile names are validated against an explicit allowlist AND a strict charset
# before they are ever used in a path, so `MIRAGE_MODEL_PROFILE=../../etc/x`
# cannot source anything outside model_profiles/.

# Explicit allowlist. A new model must be added here deliberately.
MIRAGE_MODEL_PROFILES="qwen3p5_4b qwen3p5_9b qwen3p5_27b gemma4_26b_a4b gpt_oss_20b llama31_8b"

profile_dir() {
    printf '%s\n' "${MIRAGE_REPO_ROOT}/cluster/bwunicluster3/model_profiles"
}

list_model_profiles() {
    printf '%s\n' "${MIRAGE_MODEL_PROFILES}"
}

# validate_profile_name <name>
validate_profile_name() {
    local name="${1-}" candidate
    if [[ -z "${name}" ]]; then
        printf 'no model profile given; expected one of: %s\n' "${MIRAGE_MODEL_PROFILES}" >&2
        return 1
    fi
    if [[ ! "${name}" =~ ^[a-z0-9_]+$ ]]; then
        printf 'invalid model profile name %s (allowed charset: [a-z0-9_])\n' "${name}" >&2
        return 1
    fi
    # shellcheck disable=SC2086  # allowlist is a space-separated list
    for candidate in ${MIRAGE_MODEL_PROFILES}; do
        if [[ "${name}" == "${candidate}" ]]; then
            return 0
        fi
    done
    printf 'unknown model profile %s; expected one of: %s\n' "${name}" "${MIRAGE_MODEL_PROFILES}" >&2
    return 1
}

# Fields every profile must define after being sourced.
MIRAGE_PROFILE_REQUIRED_VARS=(
    PROFILE_NAME
    MODEL_ID
    MODEL_REVISION
    SERVED_MODEL_NAME
    MODEL_FAMILY
    CV0_CONFIG_DEFAULT
    VLLM_TOOL_CALL_PARSER
    VLLM_STARTUP_TIMEOUT_S
    SBATCH_GPU_TIME_DEFAULT
    PROFILE_PARTITION_PREFERENCE
    PROFILE_MODEL_IS_GATED
)

# load_model_profile <name>
# Sources the profile and verifies its contract. Requires MIRAGE_REPO_ROOT
# (exported by lib/common.sh).
load_model_profile() {
    local name="${1-}" path var
    validate_profile_name "${name}" || return 1
    path="$(profile_dir)/${name}.sh"
    if [[ ! -f "${path}" ]]; then
        printf 'model profile file not found: %s\n' "${path}" >&2
        return 1
    fi
    # shellcheck source=/dev/null
    source "${path}"
    for var in "${MIRAGE_PROFILE_REQUIRED_VARS[@]}"; do
        if [[ -z "${!var-}" ]]; then
            printf 'model profile %s does not define %s\n' "${name}" "${var}" >&2
            return 1
        fi
    done
    if [[ "${PROFILE_NAME}" != "${name}" ]]; then
        printf 'model profile %s declares PROFILE_NAME=%s\n' "${name}" "${PROFILE_NAME}" >&2
        return 1
    fi
    if ! declare -F profile_vllm_args >/dev/null; then
        printf 'model profile %s does not define profile_vllm_args()\n' "${name}" >&2
        return 1
    fi
    # Optional hook: profiles that need a pre-launch step (e.g. resolving a chat
    # template) define profile_prepare(); provide a no-op default otherwise.
    if ! declare -F profile_prepare >/dev/null; then
        profile_prepare() { :; }
    fi
    # Optional hook: extra work for the CPU prefetch job (default: no-op).
    if ! declare -F profile_prefetch_extra >/dev/null; then
        profile_prefetch_extra() { :; }
    fi
    return 0
}
