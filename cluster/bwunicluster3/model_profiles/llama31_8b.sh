#!/usr/bin/env bash
# shellcheck shell=bash
#
# meta-llama/Llama-3.1-8B-Instruct, BF16, llama3_json tool calling.
#
# Access on Hugging Face is gated: the Llama 3.1 community license must be
# accepted with the same account whose token is in the cluster HF cache before
# the prefetch job runs.
#
# The llama3_json tool-call parser expects the official Llama 3.1 *JSON* tool-use
# chat template, which is not the template embedded in the checkpoint's
# tokenizer_config.json. It is resolved by profile_prepare() below and the run
# fails before vLLM starts if it cannot be found.

PROFILE_NAME="llama31_8b"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
SERVED_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_FAMILY="llama3.1"
PROFILE_DESCRIPTION="Llama-3.1-8B-Instruct, BF16, llama3_json tool parser + JSON tool-use template"

CV0_CONFIG_DEFAULT="configs/cv0/cv0_llama31_8b_100cp.yaml"

VLLM_TOOL_CALL_PARSER="llama3_json"
# Smallest of the three new arms, but tool-call formatting and task latency are
# not a function of parameter count alone. Start generous; tighten from the first
# measured run.
SBATCH_GPU_TIME_DEFAULT="22:00:00"
VLLM_STARTUP_TIMEOUT_S="1200"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il gpu_a100_il"
PROFILE_MODEL_IS_GATED="1"
PROFILE_GPU_MEMORY_UTILIZATION="0.85"

LLAMA31_CHAT_TEMPLATE_BASENAME="tool_chat_template_llama3.1_json.jinja"

# llama31_template_candidates
# Resolution order: explicit override, repository-controlled copy, then a
# template shipped by the installed vLLM package (several layouts are tried
# because wheels do not reliably install example files).
llama31_template_candidates() {
    local vllm_python="${VLLM_ROOT}/.venv/bin/python" site_packages=""
    if [[ -n "${LLAMA31_CHAT_TEMPLATE:-}" ]]; then
        printf '%s\n' "${LLAMA31_CHAT_TEMPLATE}"
    fi
    printf '%s\n' "${MIRAGE_REPO_ROOT}/cluster/bwunicluster3/chat_templates/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
    if [[ -x "${vllm_python}" ]]; then
        site_packages="$("${vllm_python}" -c \
            'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    fi
    if [[ -n "${site_packages}" ]]; then
        printf '%s\n' "${site_packages}/vllm/examples/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
        printf '%s\n' "${site_packages}/vllm/entrypoints/openai/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
        printf '%s\n' "$(dirname -- "${site_packages}")/../examples/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
    fi
    printf '%s\n' "${VLLM_ROOT}/examples/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
}

# profile_prepare: static existence check before the expensive vLLM launch.
profile_prepare() {
    local candidate
    LLAMA31_CHAT_TEMPLATE_RESOLVED=""
    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        if [[ -f "${candidate}" ]]; then
            LLAMA31_CHAT_TEMPLATE_RESOLVED="${candidate}"
            break
        fi
    done < <(llama31_template_candidates)

    if [[ -z "${LLAMA31_CHAT_TEMPLATE_RESOLVED}" ]]; then
        printf 'ERROR: the Llama 3.1 JSON tool-use chat template was not found.\n' >&2
        printf '  Expected (repository-controlled): %s\n' \
            "${MIRAGE_REPO_ROOT}/cluster/bwunicluster3/chat_templates/${LLAMA31_CHAT_TEMPLATE_BASENAME}" >&2
        printf '  Fetch it once on a login node with:\n' >&2
        printf '    bash cluster/bwunicluster3/fetch_chat_template.sh llama31_8b\n' >&2
        printf '  or point LLAMA31_CHAT_TEMPLATE at an existing copy.\n' >&2
        return 1
    fi

    PROFILE_CHAT_TEMPLATE_PATH="${LLAMA31_CHAT_TEMPLATE_RESOLVED}"
    if command -v sha256sum >/dev/null 2>&1; then
        PROFILE_CHAT_TEMPLATE_SHA256="$(sha256sum "${PROFILE_CHAT_TEMPLATE_PATH}" | awk '{print $1}')"
    else
        PROFILE_CHAT_TEMPLATE_SHA256="unavailable"
    fi
    printf 'Llama 3.1 chat template: %s (sha256 %s)\n' \
        "${PROFILE_CHAT_TEMPLATE_PATH}" "${PROFILE_CHAT_TEMPLATE_SHA256}"
    return 0
}

# profile_prefetch_extra: the CPU prefetch job has network access, so it also
# materialises the repository-controlled chat-template copy.
profile_prefetch_extra() {
    bash "${MIRAGE_REPO_ROOT}/cluster/bwunicluster3/fetch_chat_template.sh" "${PROFILE_NAME}"
}

profile_vllm_args() {
    VLLM_ARGS+=(
        --dtype bfloat16
        --generation-config vllm
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
        --chat-template "${PROFILE_CHAT_TEMPLATE_PATH:?profile_prepare must resolve the chat template before launch}"
    )
    # Deliberately NOT set: --gdn-prefill-backend (Qwen-only).
}
