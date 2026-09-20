#!/usr/bin/env bash

# Local vLLM port selection.
#
# Four independent one-GPU jobs can share one four-GPU H100 node, so a fixed
# port 8000 collides. The port is therefore derived deterministically from the
# Slurm job ID (so it is reproducible and appears in the logs) and then probed
# on the allocated node, advancing through a bounded range if it is taken.
#
# `port_in_use` is a separate function on purpose: tests override it to simulate
# a busy node without binding any socket (see tests/test_cluster_port.py).

# Ephemeral-port-safe window: above the usual service range, below the Linux
# default ip_local_port_range floor (32768), so we do not fight the kernel.
MIRAGE_PORT_RANGE_START="${MIRAGE_PORT_RANGE_START:-20000}"
MIRAGE_PORT_RANGE_SIZE="${MIRAGE_PORT_RANGE_SIZE:-12000}"
MIRAGE_PORT_MAX_PROBES="${MIRAGE_PORT_MAX_PROBES:-64}"

# port_in_use <port> -> 0 when something is already listening on 127.0.0.1:<port>
port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        if ss -Hltn "sport = :${port}" 2>/dev/null | grep -q .; then
            return 0
        fi
        return 1
    fi
    # Fallback: try to connect. A successful connect means someone is listening.
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# deterministic_port <job_id>
# Same job ID always yields the same starting port.
deterministic_port() {
    local job_id="$1" numeric
    numeric="${job_id//[^0-9]/}"
    [[ -n "${numeric}" ]] || numeric=0
    # 10#-prefix so a leading zero is not read as octal.
    printf '%s\n' "$(( MIRAGE_PORT_RANGE_START + (10#${numeric} % MIRAGE_PORT_RANGE_SIZE) ))"
}

# select_vllm_port <job_id> [requested_port]
# Prints the chosen port. An explicit requested port is validated and used as-is;
# otherwise the derived port is probed and advanced.
select_vllm_port() {
    local job_id="$1" requested="${2-}"
    local start port probe

    if [[ -n "${requested}" ]]; then
        if [[ ! "${requested}" =~ ^[0-9]+$ ]] || (( requested < 1024 || requested > 65535 )); then
            printf 'VLLM_PORT must be an integer in [1024, 65535], got %s\n' "${requested}" >&2
            return 1
        fi
        printf '%s\n' "${requested}"
        return 0
    fi

    start="$(deterministic_port "${job_id}")"
    for (( probe = 0; probe < MIRAGE_PORT_MAX_PROBES; probe++ )); do
        port=$(( MIRAGE_PORT_RANGE_START + ((start - MIRAGE_PORT_RANGE_START + probe) % MIRAGE_PORT_RANGE_SIZE) ))
        if ! port_in_use "${port}"; then
            printf '%s\n' "${port}"
            return 0
        fi
    done
    printf 'no free port found in [%s, %s) after %s probes\n' \
        "${MIRAGE_PORT_RANGE_START}" "$(( MIRAGE_PORT_RANGE_START + MIRAGE_PORT_RANGE_SIZE ))" \
        "${MIRAGE_PORT_MAX_PROBES}" >&2
    return 1
}
