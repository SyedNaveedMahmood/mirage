#!/usr/bin/env bash
#SBATCH --job-name=mirage-fetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=cluster/bwunicluster3/logs/%x-%j.out
#SBATCH --error=cluster/bwunicluster3/logs/%x-%j.err
#
# Shared exact-revision model prefetch for every model profile. Selected with
# MIRAGE_MODEL_PROFILE; no GPU is requested and no experiment is run here.
#
# Success is recorded by a completion marker keyed on model id AND revision, so
# a different pin re-downloads and a partial download never looks complete.

set -Eeuo pipefail

SCRIPT_DIR="${MIRAGE_SCRIPT_DIR:-$(pwd -P)/cluster/bwunicluster3}"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/profile.sh
source "${SCRIPT_DIR}/lib/profile.sh"

load_model_profile "${MIRAGE_MODEL_PROFILE:-}" \
    || die "set MIRAGE_MODEL_PROFILE to one of: $(list_model_profiles)"

VLLM_ENV="${VLLM_ROOT}/.venv"
[[ -x "${VLLM_ENV}/bin/hf" ]] || die "vLLM environment is missing; run submit_setup.sh first"

export HF_HOME="${HF_HOME:-${VLLM_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
marker_dir="${HF_HOME}/.mirage_prefetch"
marker_file="${marker_dir}/${PROFILE_NAME}@${MODEL_REVISION}.done"
lock_file="${marker_dir}/${PROFILE_NAME}.lock"
mkdir -p "${HF_HUB_CACHE}" "${marker_dir}"

printf 'Model profile : %s\n' "${PROFILE_NAME}"
printf 'Model         : %s\n' "${MODEL_ID}"
printf 'Revision      : %s\n' "${MODEL_REVISION}"
printf 'HF cache      : %s\n' "${HF_HUB_CACHE}"

if [[ -f "${marker_file}" && "${FORCE_MODEL_PREFETCH:-0}" != 1 ]]; then
    printf 'Already prefetched at this exact revision; nothing to do.\n'
    cat "${marker_file}"
    profile_prefetch_extra
    exit 0
fi

# --- authentication preflight ----------------------------------------------
# Never print, echo or log the token itself; only whether one is reachable.
have_token=0
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    have_token=1
elif [[ -f "${HF_HOME}/token" || -f "${HOME}/.cache/huggingface/token" ]]; then
    have_token=1
fi

if [[ "${have_token}" == 1 ]]; then
    printf 'Hugging Face credentials: present (value never logged)\n'
else
    printf 'Hugging Face credentials: NOT found\n'
    if [[ "${PROFILE_MODEL_IS_GATED}" == 1 ]]; then
        printf 'ERROR: %s is a GATED repository and no Hugging Face token was found.\n' "${MODEL_ID}" >&2
        printf '  1. Accept the licence at https://huggingface.co/%s with your HF account.\n' "${MODEL_ID}" >&2
        printf '  2. On a login node run:  hf auth login   (or export HF_TOKEN=<your token>)\n' >&2
        printf '     HF_HOME is %s, so the token is stored in the shared cluster cache.\n' "${HF_HOME}" >&2
        printf '  3. Re-submit the prefetch job.\n' >&2
        die "missing Hugging Face authentication for gated model ${MODEL_ID}"
    fi
fi

# --- download (exact revision only, serialised per profile) -----------------
download_log="$(mktemp)"
trap 'rm -f "${download_log}"' EXIT

run_download() {
    "${VLLM_ENV}/bin/hf" download "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --cache-dir "${HF_HUB_CACHE}" \
        2>&1 | tee "${download_log}"
    return "${PIPESTATUS[0]}"
}

download_status=0
if command -v flock >/dev/null 2>&1; then
    # Serialise concurrent prefetches of the same model against the shared cache.
    exec 9>"${lock_file}"
    flock 9
    run_download || download_status=$?
    flock -u 9
else
    run_download || download_status=$?
fi

if [[ "${download_status}" != 0 ]]; then
    if grep -Eqi '40[13]|gated|awaiting a review|access to model|restricted' "${download_log}"; then
        printf 'ERROR: access to %s was refused by the Hugging Face Hub.\n' "${MODEL_ID}" >&2
        printf '  This repository is gated. Accept its licence at\n' >&2
        printf '    https://huggingface.co/%s\n' "${MODEL_ID}" >&2
        printf '  with the SAME account whose token is configured here, wait for approval if the\n' >&2
        printf '  repository requires manual review, then re-submit this prefetch job.\n' >&2
    fi
    die "model download failed for ${MODEL_ID}@${MODEL_REVISION} (exit ${download_status})"
fi

# Newer `hf download` versions render the result as:
#   path: /absolute/snapshot/path
# Older versions may print only the path. Support both formats.
snapshot_path="$(
    awk '
        /^[[:space:]]*path:[[:space:]]*/ {
            sub(/^[[:space:]]*path:[[:space:]]*/, "")
            sub(/\r$/, "")
            path = $0
        }
        END {
            if (path != "") {
                print path
            }
        }
    ' "${download_log}"
)"
if [[ -z "${snapshot_path}" ]]; then
    snapshot_path="$(tail -n 1 "${download_log}" | tr -d '\r')"
fi
if [[ ! -d "${snapshot_path}" ]]; then
    die "download reported success but the snapshot path is not a directory: ${snapshot_path}"
fi

# Marker is written only after a verified, complete download.
cat > "${marker_file}" <<EOF
model_id=${MODEL_ID}
revision=${MODEL_REVISION}
EOF
printf '%s\n' "${snapshot_path}" > "${VLLM_ROOT}/model_snapshot_path_${PROFILE_NAME}.txt"

profile_prefetch_extra

printf 'Model ready: %s@%s at %s\n' "${MODEL_ID}" "${MODEL_REVISION}" "${snapshot_path}"
printf 'Completion marker: %s\n' "${marker_file}"
