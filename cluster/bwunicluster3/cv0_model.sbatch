#!/usr/bin/env bash
#SBATCH --job-name=cv0-model
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=22:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err
#
# Shared CV-0 job body for every model profile. It is never edited per model:
# the model identity, vLLM flags, walltime and startup budget come from
# cluster/bwunicluster3/model_profiles/<profile>.sh, selected with
# MIRAGE_MODEL_PROFILE. Submit it with submit_cv0_model.sh, which also supplies
# --job-name, --partition and --time.
#
# Resource policy (deliberate):
#   * one node, one task, one GPU, four CPU cores;
#   * no --exclusive and no --mem/--mem-per-cpu: the reference run peaked at
#     ~7.2 GB RSS with ~1.5% CPU efficiency. That 9B observation is NOT proof
#     of sufficient host RAM for larger models; check their smoke MaxRSS/OOMs;
#   * walltime below 48h so both the standard and Ice Lake GPU queues can take it.
#
# MIRAGE_JOB_MODE=smoke stops after the functional preflight (see submit_smoke_model.sh).

set -Eeuo pipefail

SCRIPT_DIR="${MIRAGE_SCRIPT_DIR:-$(pwd -P)/cluster/bwunicluster3}"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/profile.sh
source "${SCRIPT_DIR}/lib/profile.sh"
# shellcheck source=lib/port.sh
source "${SCRIPT_DIR}/lib/port.sh"

require_command curl

JOB_MODE="${MIRAGE_JOB_MODE:-production}"
[[ "${JOB_MODE}" == "production" || "${JOB_MODE}" == "smoke" ]] \
    || die "MIRAGE_JOB_MODE must be 'production' or 'smoke', got '${JOB_MODE}'"

load_model_profile "${MIRAGE_MODEL_PROFILE:-}" \
    || die "set MIRAGE_MODEL_PROFILE to one of: $(list_model_profiles)"

VLLM_VERSION_EXPECTED="${VLLM_VERSION:-0.25.1}"
VLLM_ENV="${VLLM_ROOT}/.venv"
MIRAGE_ENV="${MIRAGE_REPO_ROOT}/.venv"
CV0_CONFIG="${CV0_CONFIG:-${MIRAGE_REPO_ROOT}/${CV0_CONFIG_DEFAULT}}"

[[ -x "${VLLM_ENV}/bin/vllm" ]] || die "vLLM environment is missing; run submit_setup.sh first"
[[ -x "${MIRAGE_ENV}/bin/mirage" ]] || die "MIRAGE environment is missing; run submit_setup.sh first"
[[ -f "${CV0_CONFIG}" ]] || die "CV-0 config not found: ${CV0_CONFIG}"
# Re-check after queueing as well: YAML lives in the shared checkout.
"${MIRAGE_ENV}/bin/python" "${SCRIPT_DIR}/lib/validate_profile_config.py" \
    "${CV0_CONFIG}" "${MODEL_ID}" "${MODEL_REVISION}" "${SERVED_MODEL_NAME}" "${MODEL_FAMILY}"

job_id="${SLURM_JOB_ID:-manual}"
job_tag="${PROFILE_NAME}-${job_id}"

export HF_HOME="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VLLM_ROOT}/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${VLLM_ROOT}/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${VLLM_ROOT}/cache/triton}"
export OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export MPLBACKEND=Agg

job_artifacts="${MIRAGE_REPO_ROOT}/runs/_cluster_jobs/${job_tag}"
mkdir -p "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${VLLM_ROOT}/logs" "${SCRIPT_DIR}/logs" "${job_artifacts}"

vllm_version="$("${VLLM_ENV}/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
[[ "${vllm_version}" == "${VLLM_VERSION_EXPECTED}" ]] \
    || die "vLLM ${vllm_version} is installed; expected ${VLLM_VERSION_EXPECTED}"
export VLLM_SERVER_VERSION="${vllm_version}"

# --- port selection ---------------------------------------------------------
# Four one-GPU jobs can share one four-GPU node, so port 8000 is not assumed free.
selected_port="$(select_vllm_port "${job_id}" "${VLLM_PORT:-}")" \
    || die "could not select a free local port for vLLM"
export VLLM_PORT="${selected_port}"
export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

# --- model-specific pre-launch checks (e.g. chat template) ------------------
profile_prepare || die "profile ${PROFILE_NAME} failed its pre-launch checks"

server_log="${VLLM_ROOT}/logs/vllm-${job_tag}.log"
models_json="${job_artifacts}/v1_models.json"
metadata_json="${job_artifacts}/cluster_job_metadata.json"
server_pid=""

cleanup() {
    local status=$?
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    # Always preserve the server log with the job artifacts, even on failure.
    if [[ -f "${server_log}" ]]; then
        cp -f "${server_log}" "${job_artifacts}/vllm-server.log" 2>/dev/null || true
    fi
    return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${MIRAGE_REPO_ROOT}"

printf '=== MIRAGE-Persist CV-0 cluster job =====================================\n'
printf 'mode              : %s\n' "${JOB_MODE}"
printf 'job id            : %s\n' "${job_id}"
printf 'job name          : %s\n' "${SLURM_JOB_NAME:-unknown}"
printf 'partition         : %s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'nodelist          : %s\n' "${SLURM_JOB_NODELIST:-unknown}"
printf 'hostname          : %s\n' "$(hostname)"
printf 'cpus per task     : %s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
printf 'gpus              : %s\n' "${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-unknown}}"
printf 'model profile     : %s (%s)\n' "${PROFILE_NAME}" "${PROFILE_DESCRIPTION:-}"
printf 'model             : %s@%s\n' "${MODEL_ID}" "${MODEL_REVISION}"
printf 'tool-call parser  : %s\n' "${VLLM_TOOL_CALL_PARSER}"
printf 'vLLM version      : %s\n' "${vllm_version}"
printf 'vLLM port         : %s\n' "${VLLM_PORT}"
printf 'CV-0 config       : %s\n' "${CV0_CONFIG}"
printf 'job artifacts     : %s\n' "${job_artifacts}"
printf '=========================================================================\n'

# Slurm/GPU facts available only from inside the allocation.
scontrol show job "${job_id}" > "${job_artifacts}/scontrol_show_job.txt" 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${job_artifacts}/nvidia-smi.txt" 2>&1 || true
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
        --format=csv,noheader > "${job_artifacts}/gpu_identity.csv" 2>/dev/null || true
    cat "${job_artifacts}/nvidia-smi.txt"
fi

# --- vLLM launch ------------------------------------------------------------
# Built as an array: no eval, no word-splitting surprises in model paths.
VLLM_ARGS=(
    serve "${MODEL_ID}"
    --revision "${MODEL_REVISION}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 127.0.0.1
    --port "${VLLM_PORT}"
    --api-key "${OPENAI_COMPATIBLE_API_KEY}"
    --tensor-parallel-size 1
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-${PROFILE_GPU_MEMORY_UTILIZATION:-0.85}}"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}"
    # The MIRAGE client is synchronous: one in-flight sequence is all it uses.
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}"
    --download-dir "${HF_HUB_CACHE}"
)
# Prefix caching is requested explicitly rather than left to the engine default
# (continuations from one checkpoint share a long prefix). VLLM_ENABLE_PREFIX_CACHING=0
# drops the flag for an engine build that does not accept it.
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-1}" == 1 ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
fi
profile_vllm_args

# Redacted copy for the log/provenance: never print the API key.
redacted_cmd=()
skip_next=0
for arg in "${VLLM_ENV}/bin/vllm" "${VLLM_ARGS[@]}"; do
    if [[ "${skip_next}" == 1 ]]; then
        redacted_cmd+=("***")
        skip_next=0
        continue
    fi
    if [[ "${arg}" == "--api-key" ]]; then
        skip_next=1
    fi
    redacted_cmd+=("${arg}")
done
printf 'vLLM launch: %s\n' "${redacted_cmd[*]}"
printf '%s\n' "${redacted_cmd[*]}" > "${job_artifacts}/vllm_launch_command.txt"

startup_begin="$(date +%s)"
"${VLLM_ENV}/bin/vllm" "${VLLM_ARGS[@]}" >"${server_log}" 2>&1 &
server_pid=$!

# --- health polling ---------------------------------------------------------
# Connection refusals are EXPECTED while the server loads weights, so curl is
# fully silenced and progress is reported periodically instead of per attempt.
startup_timeout_s="${VLLM_STARTUP_TIMEOUT_S}"
poll_interval_s=5
max_polls=$(( startup_timeout_s / poll_interval_s ))
progress_every=12   # ~60 s
ready=0
for (( poll = 1; poll <= max_polls; poll++ )); do
    if curl --silent --output /dev/null --fail --max-time 5 \
        -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" \
        "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        printf 'vLLM exited during startup. Last 120 log lines:\n' >&2
        tail -n 120 "${server_log}" >&2 || true
        exit 1
    fi
    if (( poll % progress_every == 0 )); then
        printf 'waiting for vLLM: %ss elapsed of %ss budget (pid %s)\n' \
            "$(( poll * poll_interval_s ))" "${startup_timeout_s}" "${server_pid}"
    fi
    sleep "${poll_interval_s}"
done

if [[ "${ready}" != 1 ]]; then
    printf 'vLLM did not become healthy within %ss. Last 120 log lines:\n' "${startup_timeout_s}" >&2
    tail -n 120 "${server_log}" >&2 || true
    die "vLLM startup timed out; full log: ${server_log}"
fi
startup_duration_s=$(( $(date +%s) - startup_begin ))
printf 'vLLM healthy after %ss\n' "${startup_duration_s}"

# --- served-model verification ---------------------------------------------
curl --silent --show-error --fail --max-time 30 \
    -H "Authorization: Bearer ${OPENAI_COMPATIBLE_API_KEY}" \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" > "${models_json}" \
    || die "could not read /v1/models from the local server"
if ! grep -Fq "\"${SERVED_MODEL_NAME}\"" "${models_json}"; then
    printf '/v1/models did not report %s:\n' "${SERVED_MODEL_NAME}" >&2
    cat "${models_json}" >&2
    die "served model mismatch"
fi
printf '/v1/models reports the expected served model: %s\n' "${SERVED_MODEL_NAME}"

# --- run/job provenance -----------------------------------------------------
MIRAGE_META_STARTUP_S="${startup_duration_s}" \
MIRAGE_META_PROFILE="${PROFILE_NAME}" \
MIRAGE_META_PROFILE_DESC="${PROFILE_DESCRIPTION:-}" \
MIRAGE_META_MODEL_ID="${MODEL_ID}" \
MIRAGE_META_MODEL_REVISION="${MODEL_REVISION}" \
MIRAGE_META_SERVED_MODEL="${SERVED_MODEL_NAME}" \
MIRAGE_META_MODEL_FAMILY="${MODEL_FAMILY}" \
MIRAGE_META_TOOL_PARSER="${VLLM_TOOL_CALL_PARSER}" \
MIRAGE_META_REASONING_PARSER="${VLLM_REASONING_PARSER:-}" \
MIRAGE_META_REASONING_EFFORT="${GPT_OSS_REASONING_EFFORT:-}" \
MIRAGE_META_CHAT_TEMPLATE="${PROFILE_CHAT_TEMPLATE_PATH:-}" \
MIRAGE_META_CHAT_TEMPLATE_SHA256="${PROFILE_CHAT_TEMPLATE_SHA256:-}" \
MIRAGE_META_VLLM_VERSION="${vllm_version}" \
MIRAGE_META_PORT="${VLLM_PORT}" \
MIRAGE_META_CONFIG="${CV0_CONFIG}" \
MIRAGE_META_MODE="${JOB_MODE}" \
MIRAGE_META_LAUNCH_CMD="${redacted_cmd[*]}" \
MIRAGE_META_MODELS_JSON="${models_json}" \
    "${MIRAGE_ENV}/bin/python" "${SCRIPT_DIR}/lib/write_job_metadata.py" "${metadata_json}"
printf 'Wrote job metadata: %s\n' "${metadata_json}"

# --- functional API preflight ----------------------------------------------
# A handful of tokens against the already-running server: plain completion, one
# JSON tool schema, and tool-call parsing through the same backend MIRAGE uses.
# It must fail here, not eight hours into the campaign.
printf 'Running functional API preflight...\n'
preflight_status=0
"${MIRAGE_ENV}/bin/mirage" preflight --config "${CV0_CONFIG}" \
    --json-out "${job_artifacts}/preflight.json" || preflight_status=$?
if [[ "${preflight_status}" != 0 ]]; then
    printf 'Functional preflight FAILED (exit %s); not starting CV-0.\n' "${preflight_status}" >&2
    tail -n 60 "${server_log}" >&2 || true
    exit "${preflight_status}"
fi
printf 'Functional API preflight passed.\n'

if [[ "${JOB_MODE}" == "smoke" ]]; then
    printf 'Smoke mode: model load, health, generation and one tool call verified.\n'
    printf 'NOT running the CV-0 campaign. Artifacts: %s\n' "${job_artifacts}"
    exit 0
fi

# --- CV-0 -------------------------------------------------------------------
mirage_cmd=("${MIRAGE_ENV}/bin/mirage" cv0 run --config "${CV0_CONFIG}")
printf 'MIRAGE command: %s\n' "${mirage_cmd[*]}"
printf '%s\n' "${mirage_cmd[*]}" > "${job_artifacts}/mirage_command.txt"

set +e
"${mirage_cmd[@]}"
cv0_status=$?
set -e
printf 'CV-0 process exit code: %s (0=PASS, 1=FAIL, 2=KILL)\n' "${cv0_status}"
printf '%s\n' "${cv0_status}" > "${job_artifacts}/cv0_exit_code.txt"

# Mirror the server log and job metadata into the experiment's own run directory
# so the scientific artifact set is self-contained.
experiment_id="$("${MIRAGE_ENV}/bin/python" -c '
import sys
from mirage_persist.config.loader import resolve_config
print(resolve_config(sys.argv[1]).get("experiment_id", ""))
' "${CV0_CONFIG}" 2>/dev/null || true)"
if [[ -n "${experiment_id}" && -d "${MIRAGE_REPO_ROOT}/runs/${experiment_id}" ]]; then
    latest_run="$(find "${MIRAGE_REPO_ROOT}/runs/${experiment_id}" -mindepth 1 -maxdepth 1 -type d \
        | sort | tail -n 1)"
    if [[ -n "${latest_run}" ]]; then
        cp -f "${server_log}" "${latest_run}/vllm-server.log" 2>/dev/null || true
        cp -f "${metadata_json}" "${latest_run}/cluster_job_metadata.json" 2>/dev/null || true
        cp -f "${models_json}" "${latest_run}/v1_models.json" 2>/dev/null || true
        printf 'Copied server log and job metadata into %s\n' "${latest_run}"
    fi
fi

exit "${cv0_status}"
