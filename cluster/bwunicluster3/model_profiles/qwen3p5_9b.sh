#!/usr/bin/env bash
# shellcheck shell=bash
#
# Qwen3.5-9B -- the reference arm. The first successful 100-checkpoint run took
# ~16h52m on one H100 with this configuration.

PROFILE_NAME="qwen3p5_9b"
MODEL_ID="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SERVED_MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_FAMILY="qwen3.5"
PROFILE_DESCRIPTION="Qwen3.5-9B dense, BF16, non-thinking, qwen3_xml tool parser"

CV0_CONFIG_DEFAULT="configs/cv0/cv0_qwen3p5_9b_100cp.yaml"

VLLM_TOOL_CALL_PARSER="qwen3_xml"
# Measured: ~16h52m end to end. 22h leaves headroom and stays under the 48h
# ceiling, so the job remains eligible for both the standard and Ice Lake queues.
SBATCH_GPU_TIME_DEFAULT="22:00:00"
VLLM_STARTUP_TIMEOUT_S="1200"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il gpu_a100_il"
PROFILE_MODEL_IS_GATED="0"
PROFILE_GPU_MEMORY_UTILIZATION="0.85"

# profile_vllm_args: append model-specific flags to the VLLM_ARGS array.
profile_vllm_args() {
    VLLM_ARGS+=(
        --dtype bfloat16
        --generation-config vllm
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
        --default-chat-template-kwargs '{"enable_thinking": false}'
        # Qwen-specific: the GDN prefill backend. Do not copy this to other families.
        --gdn-prefill-backend triton
        # CV-0 sends text only; avoid reserving vision encoder/profiling memory.
        --limit-mm-per-prompt '{"image": 0, "video": 0}'
    )
}
