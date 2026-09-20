#!/usr/bin/env bash
# Text-only CV-0; same scientific sampling and serving flags as the 9B arm.
PROFILE_NAME="qwen3p5_4b"
MODEL_ID="Qwen/Qwen3.5-4B"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERVED_MODEL_NAME="Qwen/Qwen3.5-4B"
MODEL_FAMILY="qwen3.5"
PROFILE_DESCRIPTION="Qwen3.5-4B dense, BF16, non-thinking, qwen3_xml tool parser"
CV0_CONFIG_DEFAULT="configs/cv0/cv0_qwen3p5_4b_100cp.yaml"
VLLM_TOOL_CALL_PARSER="qwen3_xml"
# Planning budget, not a measured runtime.
SBATCH_GPU_TIME_DEFAULT="22:00:00"
VLLM_STARTUP_TIMEOUT_S="1200"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il gpu_a100_il"
PROFILE_MODEL_IS_GATED="0"
PROFILE_GPU_MEMORY_UTILIZATION="0.85"

profile_vllm_args() {
    VLLM_ARGS+=(
        --dtype bfloat16
        --generation-config vllm
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
        --default-chat-template-kwargs '{"enable_thinking": false}'
        --gdn-prefill-backend triton
        --limit-mm-per-prompt '{"image": 0, "video": 0}'
    )
}
