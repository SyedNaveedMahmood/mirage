#!/usr/bin/env bash
# Materialise a repository-controlled copy of a model's chat template.
#
# Run this once on a login node (it needs network access, no GPU); the CPU
# prefetch job also calls it. It downloads the template that ships with the
# pinned vLLM release, records its SHA-256 and source URL next to it, and
# refuses to overwrite a previously recorded digest -- so once a template has
# been fetched (and ideally committed), later fetches can only confirm it.
#
# Usage: bash cluster/bwunicluster3/fetch_chat_template.sh llama31_8b
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/profile.sh
source "${SCRIPT_DIR}/lib/profile.sh"

profile_name="${1:-}"
load_model_profile "${profile_name}" || die "usage: $0 <model-profile>"

if [[ -z "${LLAMA31_CHAT_TEMPLATE_BASENAME:-}" ]]; then
    printf 'Model profile %s does not use an external chat template; nothing to do.\n' "${profile_name}"
    exit 0
fi

VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
template_dir="${SCRIPT_DIR}/chat_templates"
template_path="${template_dir}/${LLAMA31_CHAT_TEMPLATE_BASENAME}"
digest_path="${template_path}.sha256"
provenance_path="${template_path}.provenance.json"
source_url="${CHAT_TEMPLATE_SOURCE_URL:-https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VERSION}/examples/${LLAMA31_CHAT_TEMPLATE_BASENAME}}"

mkdir -p "${template_dir}"

if [[ -f "${template_path}" && "${FORCE_CHAT_TEMPLATE_REFETCH:-0}" != 1 ]]; then
    printf 'Chat template already present: %s\n' "${template_path}"
    if [[ -f "${digest_path}" ]] && command -v sha256sum >/dev/null 2>&1; then
        (cd "${template_dir}" && sha256sum --check --status "$(basename -- "${digest_path}")") \
            || die "chat template does not match its recorded digest: ${digest_path}"
        printf 'Digest verified against %s\n' "${digest_path}"
    fi
    exit 0
fi

require_command curl
tmp_file="$(mktemp)"
trap 'rm -f "${tmp_file}"' EXIT

printf 'Downloading %s\n' "${source_url}"
curl --silent --show-error --fail --location --retry 3 --max-time 120 \
    --output "${tmp_file}" "${source_url}" \
    || die "download failed: ${source_url}"
[[ -s "${tmp_file}" ]] || die "downloaded chat template is empty: ${source_url}"
grep -q '{%' "${tmp_file}" || die "downloaded file does not look like a Jinja template: ${source_url}"

new_digest="$(sha256sum "${tmp_file}" | awk '{print $1}')"
if [[ -f "${digest_path}" ]]; then
    recorded="$(awk '{print $1}' "${digest_path}")"
    [[ "${recorded}" == "${new_digest}" ]] \
        || die "downloaded template digest ${new_digest} does not match the pinned ${recorded} in ${digest_path}"
fi

install -m 0644 "${tmp_file}" "${template_path}"
printf '%s  %s\n' "${new_digest}" "${LLAMA31_CHAT_TEMPLATE_BASENAME}" > "${digest_path}"
cat > "${provenance_path}" <<JSON
{
  "template": "${LLAMA31_CHAT_TEMPLATE_BASENAME}",
  "model_profile": "${profile_name}",
  "model_id": "${MODEL_ID}",
  "source_url": "${source_url}",
  "vllm_version": "${VLLM_VERSION}",
  "sha256": "${new_digest}",
  "fetched_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

printf 'Chat template ready: %s\n' "${template_path}"
printf 'sha256: %s\n' "${new_digest}"
printf 'Commit %s, %s and %s so the template becomes repository-controlled.\n' \
    "${template_path}" "${digest_path}" "${provenance_path}"
