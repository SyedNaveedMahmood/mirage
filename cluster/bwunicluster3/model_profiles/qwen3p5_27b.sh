#!/usr/bin/env bash
# ~51.75 GiB of checkpoint tensors alone. Use H100 production partitions;
# no 40-GiB GPUs. Allocation fit and runtime still require a cluster smoke.
PROFILE_NAME="qwen3p5_27b"
MODEL_ID="Qwen/Qwen3.5-27B"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
SERVED_MODEL_NAME="Qwen/Qwen3.5-27B"
MODEL_FAMILY="qwen3.5"
PROFILE_DESCRIPTION="Qwen3.5-27B dense, BF16, non-thinking, qwen3_xml tool parser"
CV0_CONFIG_DEFAULT="configs/cv0/cv0_qwen3p5_27b_100cp.yaml"
VLLM_TOOL_CALL_PARSER="qwen3_xml"
# Unmeasured upper planning budget; a 100-checkpoint run may still time out.
SBATCH_GPU_TIME_DEFAULT="47:00:00"
VLLM_STARTUP_TIMEOUT_S="1800"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il"
PROFILE_MODEL_IS_GATED="0"
PROFILE_GPU_MEMORY_UTILIZATION="0.90"

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
