#!/usr/bin/env bash

# Queue-aware GPU partition selection for bwUniCluster 3.0.
#
# `sinfo_t_idle` is a site wrapper whose exact output format is not contractual,
# so the parser below is deliberately tolerant: it looks for a line that mentions
# a candidate partition as a whole field and takes the first integer on that line
# as the idle-node count. Anything it cannot understand degrades to "no idle
# information", never to a wrong partition.
#
# These functions are pure text -> decision; `select_gpu_partition` never calls
# sinfo itself, which is what makes it testable off-cluster
# (see tests/test_cluster_partition.py).

# Compatible regular (non-development) GPU partitions, in default preference
# order. H100 first, then Ice Lake H100, then Ice Lake A100.
MIRAGE_DEFAULT_GPU_PARTITIONS="gpu_h100 gpu_h100_il gpu_a100_il"

# Development partitions must never be selected automatically for a production run.
MIRAGE_DEV_PARTITION_PATTERN='^dev_'


# slurm_partition_list <candidates>
# Convert the reviewed space-separated profile compatibility set to the
# comma-separated syntax accepted by sbatch --partition. This deliberately does
# not inspect current idle nodes: scheduling availability is Slurm's job.
slurm_partition_list() {
    local candidates="$1" partition out=""
    # shellcheck disable=SC2086  # candidates is a reviewed space-separated list
    for partition in ${candidates}; do
        if [[ -z "${out}" ]]; then
            out="${partition}"
        else
            out="${out},${partition}"
        fi
    done
    [[ -n "${out}" ]] || return 1
    printf '%s\n' "${out}"
}

# idle_nodes_for_partition <partition> <text>
# Prints the idle-node count found for <partition>, or -1 when the text carries
# no usable information for it.
idle_nodes_for_partition() {
    local partition="$1" text="$2"
    local line field count found=-1
    while IFS= read -r line; do
        # Strip common separators so "gpu_h100:" and "gpu_h100," still match as fields.
        local normalized="${line//[:,|]/ }"
        local matched=0
        # shellcheck disable=SC2086  # word splitting is the point: iterate fields
        for field in ${normalized}; do
            if [[ "${field}" == "${partition}" ]]; then
                matched=1
                break
            fi
        done
        [[ "${matched}" == 1 ]] || continue
        count=""
        # shellcheck disable=SC2086  # word splitting is the point: iterate fields
        for field in ${normalized}; do
            if [[ "${field}" == "${partition}" ]]; then
                continue
            fi
            if [[ "${field}" =~ ^[0-9]+$ ]]; then
                count="${field}"
                break
            fi
        done
        if [[ -n "${count}" ]]; then
            found="${count}"
        elif [[ "${found}" == -1 ]]; then
            # The partition is named but no count is parseable: known, unknown load.
            found=0
        fi
        break
    done <<<"${text}"
    printf '%s\n' "${found}"
}

# select_gpu_partition <candidates> [sinfo_text]
# Prints "<partition>\t<reason>". <candidates> is a space-separated preference
# list; the first candidate is the deterministic default when nothing is idle.
select_gpu_partition() {
    local candidates="$1" text="${2-}"
    local default_partition="${candidates%% *}"
    local partition idle any_info=0

    [[ -n "${default_partition}" ]] || {
        printf '%s\t%s\n' "" "no candidate partitions configured"
        return 1
    }

    if [[ -z "${text//[[:space:]]/}" ]]; then
        printf '%s\t%s\n' "${default_partition}" \
            "no idle-node information available; deterministic default (first compatible partition)"
        return 0
    fi

    # shellcheck disable=SC2086  # candidates is a space-separated preference list
    for partition in ${candidates}; do
        idle="$(idle_nodes_for_partition "${partition}" "${text}")"
        [[ "${idle}" == -1 ]] || any_info=1
        if [[ "${idle}" != -1 && "${idle}" -gt 0 ]]; then
            printf '%s\t%s\n' "${partition}" \
                "${idle} idle node(s) reported; highest-preference compatible partition with idle capacity"
            return 0
        fi
    done

    if [[ "${any_info}" == 1 ]]; then
        printf '%s\t%s\n' "${default_partition}" \
            "no compatible partition reports idle nodes; deterministic default (first compatible partition)"
    else
        printf '%s\t%s\n' "${default_partition}" \
            "idle-node output was not parseable; deterministic default (first compatible partition)"
    fi
    return 0
}

# validate_production_partition <partition> <candidates>
# Rejects development queues and unknown partitions for production submissions.
validate_production_partition() {
    local partition="$1" candidates="$2" candidate
    if [[ "${partition}" =~ ${MIRAGE_DEV_PARTITION_PATTERN} ]]; then
        printf 'development partition %s is not allowed for a production CV-0 run\n' "${partition}" >&2
        return 1
    fi
    # shellcheck disable=SC2086  # candidates is a space-separated preference list
    for candidate in ${candidates}; do
        if [[ "${partition}" == "${candidate}" ]]; then
            return 0
        fi
    done
    printf 'partition %s is not in the compatible set: %s\n' "${partition}" "${candidates}" >&2
    return 1
}

# read_sinfo_t_idle
# Calls the site helper exactly once and never fails the caller. Cluster-only;
# on any error (missing command, non-zero exit) it prints nothing.
read_sinfo_t_idle() {
    command -v sinfo_t_idle >/dev/null 2>&1 || return 0
    sinfo_t_idle 2>/dev/null || true
}
