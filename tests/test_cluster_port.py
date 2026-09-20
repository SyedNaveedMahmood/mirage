"""Deterministic vLLM port selection (cluster/bwunicluster3/lib/port.sh).

No socket is ever bound and no privileged port is touched: `port_in_use` is
overridden after sourcing the library, which is exactly why it is a separate
function there.
"""

from __future__ import annotations

import subprocess

import pytest

from shell_support import CLUSTER_DIR, find_bash

_LIB = CLUSTER_DIR / "lib" / "port.sh"
_BASH = find_bash()

pytestmark = pytest.mark.skipif(_BASH is None, reason="a POSIX bash is required for shell-library tests")

RANGE_START = 20000
RANGE_SIZE = 12000


def _run(job_id: str, requested: str = "", busy: str = "") -> subprocess.CompletedProcess[str]:
    script = f"""
set -Eeuo pipefail
source "{_LIB.as_posix()}"
# Test double: the ports in BUSY are treated as already listening.
port_in_use() {{
    local candidate="$1" busy_port
    for busy_port in ${{BUSY}}; do
        if [[ "${{candidate}}" == "${{busy_port}}" ]]; then
            return 0
        fi
    done
    return 1
}}
select_vllm_port "$1" "$2"
"""
    return subprocess.run(
        [_BASH, "-c", script, "bash", job_id, requested],
        capture_output=True,
        text=True,
        env={"BUSY": busy, "PATH": "/usr/bin:/bin"},
    )


def test_port_is_deterministic_for_a_job_id():
    first = _run("1234567").stdout.strip()
    second = _run("1234567").stdout.strip()
    assert first == second
    assert first.isdigit()


def test_port_is_in_the_expected_unprivileged_range():
    port = int(_run("987654").stdout.strip())
    assert RANGE_START <= port < RANGE_START + RANGE_SIZE
    assert port != 8000  # the old fixed port is never assumed free


def test_different_jobs_on_one_node_get_different_ports():
    ports = {_run(str(job_id)).stdout.strip() for job_id in (5000001, 5000002, 5000003, 5000004)}
    assert len(ports) == 4


def test_busy_port_advances_within_the_range():
    free = _run("1234567").stdout.strip()
    taken = _run("1234567", busy=free).stdout.strip()
    assert taken != free
    assert int(taken) == int(free) + 1


def test_advances_past_a_run_of_busy_ports():
    free = int(_run("1234567").stdout.strip())
    busy = " ".join(str(free + offset) for offset in range(5))
    assert int(_run("1234567", busy=busy).stdout.strip()) == free + 5


def test_explicit_request_is_honoured():
    assert _run("1234567", requested="8123").stdout.strip() == "8123"


def test_explicit_request_is_validated():
    for bad in ("80", "not-a-port", "70000"):
        result = _run("1234567", requested=bad)
        assert result.returncode != 0, bad
        assert "VLLM_PORT" in result.stderr


def test_non_numeric_job_id_still_yields_a_port():
    port = int(_run("manual").stdout.strip())
    assert RANGE_START <= port < RANGE_START + RANGE_SIZE


def test_leading_zero_job_id_is_not_read_as_octal():
    result = _run("0891234")
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= RANGE_START


def test_exhausted_range_fails_loudly():
    free = int(_run("1234567").stdout.strip())
    busy = " ".join(str(free + offset) for offset in range(80))
    result = _run("1234567", busy=busy)
    assert result.returncode != 0
    assert "no free port" in result.stderr
