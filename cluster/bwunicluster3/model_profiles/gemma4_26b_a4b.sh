#!/usr/bin/env bash
# shellcheck shell=bash
#
# Gemma 4 26B-A4B (MoE, ~4B active parameters), BF16, non-thinking.
# Access on Hugging Face is gated: the license must be accepted with the same
# account whose token is in the cluster HF cache before the prefetch job runs.

PROFILE_NAME="gemma4_26b_a4b"
MODEL_ID="google/gemma-4-26B-A4B-it"
MODEL_REVISION="462a98a12e28e2cbcfccaf78fe41e3e50235e6ae"
SERVED_MODEL_NAME="google/gemma-4-26B-A4B-it"
MODEL_FAMILY="gemma4"
PROFILE_DESCRIPTION="Gemma 4 26B-A4B MoE, BF16, non-thinking, gemma4 tool parser"

CV0_CONFIG_DEFAULT="configs/cv0/cv0_gemma4_26b_a4b_100cp.yaml"

VLLM_TOOL_CALL_PARSER="gemma4"
# 26B total parameters: weight load and MoE graph capture are slower than the
# 9B reference arm, and per-decision latency is not predictable from the ~4B
# active count alone. Start generous; tighten from the first measured run.
SBATCH_GPU_TIME_DEFAULT="30:00:00"
VLLM_STARTUP_TIMEOUT_S="2400"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il gpu_a100_il"
PROFILE_MODEL_IS_GATED="1"
# 26B BF16 weights need most of an 80 GB card; 0.90 leaves room for the KV cache
# at 32K context while keeping a safety margin against fragmentation.
PROFILE_GPU_MEMORY_UTILIZATION="0.90"

# CV-0 is a text-only protocol. Disabling the multimodal input paths avoids
# allocating vision/audio encoders and their profiling buffers. Set
# VLLM_LIMIT_MM_PER_PROMPT="" to drop the flag if the installed vLLM rejects it.
if [[ -z "${VLLM_LIMIT_MM_PER_PROMPT+set}" ]]; then
    VLLM_LIMIT_MM_PER_PROMPT='{"image": 0, "video": 0, "audio": 0}'
fi

profile_vllm_args() {
    VLLM_ARGS+=(
        --dtype bfloat16
        --generation-config vllm
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
        --default-chat-template-kwargs '{"enable_thinking": false}'
    )
    if [[ -n "${VLLM_LIMIT_MM_PER_PROMPT}" ]]; then
        VLLM_ARGS+=(--limit-mm-per-prompt "${VLLM_LIMIT_MM_PER_PROMPT}")
    fi
    # Deliberately NOT set: --gdn-prefill-backend (Qwen-only).
}
