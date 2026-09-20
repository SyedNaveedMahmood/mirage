#!/usr/bin/env bash
# shellcheck shell=bash
#
# openai/gpt-oss-20b on its native MXFP4 path.
#
# Deliberately NOT set: --dtype and --quantization. gpt-oss ships MXFP4 weights
# and vLLM selects that path from the checkpoint itself; forcing bfloat16 would
# either upcast the weights (changing the model under test and the memory
# profile) or fail outright.

PROFILE_NAME="gpt_oss_20b"
MODEL_ID="openai/gpt-oss-20b"
MODEL_REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"
SERVED_MODEL_NAME="openai/gpt-oss-20b"
MODEL_FAMILY="gpt-oss"
PROFILE_DESCRIPTION="gpt-oss-20b, native MXFP4, Harmony, openai tool parser, reasoning_effort=low"

CV0_CONFIG_DEFAULT="configs/cv0/cv0_gpt_oss_20b_100cp.yaml"

VLLM_TOOL_CALL_PARSER="openai"
# Harmony reasoning emits extra tokens even at reasoning_effort=low, so decisions
# are slower than the parameter count suggests. Start generous; tighten from the
# first measured run.
SBATCH_GPU_TIME_DEFAULT="30:00:00"
VLLM_STARTUP_TIMEOUT_S="1800"
PROFILE_PARTITION_PREFERENCE="gpu_h100 gpu_h100_il gpu_a100_il"
PROFILE_MODEL_IS_GATED="0"
PROFILE_GPU_MEMORY_UTILIZATION="0.90"

# Harmony reasoning parser: keeps chain-of-thought in `reasoning_content` instead
# of leaking it into `content`, which is what the MIRAGE action classifier reads.
# Set VLLM_REASONING_PARSER="" to drop the flag.
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER-openai_gptoss}"

# Reasoning effort for this experiment. Sent by MIRAGE on every request through
# configs/models/openai_compat_vllm_gpt_oss_20b.yaml; also set as the server
# default so the two cannot silently diverge.
GPT_OSS_REASONING_EFFORT="${GPT_OSS_REASONING_EFFORT:-low}"

profile_vllm_args() {
    VLLM_ARGS+=(
        --generation-config vllm
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
        --default-chat-template-kwargs "{\"reasoning_effort\": \"${GPT_OSS_REASONING_EFFORT}\"}"
    )
    if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
        VLLM_ARGS+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
    fi
    # Deliberately NOT set: --dtype, --quantization, --gdn-prefill-backend.
}
