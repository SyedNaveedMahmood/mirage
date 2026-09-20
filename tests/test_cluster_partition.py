"""Queue-aware partition selection (cluster/bwunicluster3/lib/partition.sh).

The parser is exercised with stored fixture strings only: `sinfo_t_idle` is never
invoked, so these tests run anywhere. The fixtures cover the plausible shapes of
that site wrapper's output, because its format is not contractual.
"""

from __future__ import annotations

import subprocess

import pytest

from shell_support import CLUSTER_DIR, find_bash

_LIB = CLUSTER_DIR / "lib" / "partition.sh"
_BASH = find_bash()

pytestmark = pytest.mark.skipif(_BASH is None, reason="a POSIX bash is required for shell-library tests")

CANDIDATES = "gpu_h100 gpu_h100_il gpu_a100_il"

# --- fixtures: plausible `sinfo_t_idle` outputs ---------------------------- #
H100_IDLE = """\
Partition        Idle nodes
dev_gpu_h100     2
gpu_h100         5
gpu_h100_il      3
gpu_a100_il      1
cpu              120
"""

ONLY_IL_H100_IDLE = """\
Partition        Idle nodes
gpu_h100         0
gpu_h100_il      4
gpu_a100_il      0
"""

ONLY_A100_IDLE = """\
Partition        Idle nodes
gpu_h100         0
gpu_h100_il      0
gpu_a100_il      2
"""

NOTHING_IDLE = """\
Partition        Idle nodes
gpu_h100         0
gpu_h100_il      0
gpu_a100_il      0
"""

ONLY_DEV_IDLE = """\
Partition        Idle nodes
dev_gpu_h100     4
dev_gpu_a100_il  2
gpu_h100         0
gpu_h100_il      0
gpu_a100_il      0
"""

ALTERNATIVE_FORMAT = """\
Partition gpu_h100 : 0 nodes idle
Partition gpu_h100_il : 7 nodes idle
Partition gpu_a100_il : 0 nodes idle
"""

MALFORMED = """\
sinfo_t_idle: unexpected error while querying the controller
<<<garbage>>>
"""

UNAVAILABLE = ""


def _select(text: str, candidates: str = CANDIDATES) -> tuple[str, str]:
    script = f'set -Eeuo pipefail; source "{_LIB.as_posix()}"; select_gpu_partition "$1" "$2"'
    out = subprocess.run(
        [_BASH, "-c", script, "bash", candidates, text],
        capture_output=True,
        text=True,
        check=True,
    )
    partition, _, reason = out.stdout.strip().partition("\t")
    return partition, reason


def _validate(partition: str, candidates: str = CANDIDATES) -> int:
    script = f'source "{_LIB.as_posix()}"; validate_production_partition "$1" "$2"'
    return subprocess.run(
        [_BASH, "-c", script, "bash", partition, candidates],
        capture_output=True,
        text=True,
    ).returncode


def _slurm_partition_list(candidates: str = CANDIDATES) -> str:
    script = f'set -Eeuo pipefail; source "{_LIB.as_posix()}"; slurm_partition_list "$1"'
    out = subprocess.run(
        [_BASH, "-c", script, "bash", candidates],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_slurm_partition_list_offers_all_compatible_partitions():
    assert _slurm_partition_list() == "gpu_h100,gpu_h100_il,gpu_a100_il"


def test_prefers_h100_when_idle():
    partition, reason = _select(H100_IDLE)
    assert partition == "gpu_h100"
    assert "idle" in reason


def test_only_ice_lake_h100_idle():
    assert _select(ONLY_IL_H100_IDLE)[0] == "gpu_h100_il"


def test_only_a100_idle():
    assert _select(ONLY_A100_IDLE)[0] == "gpu_a100_il"


def test_nothing_idle_uses_deterministic_default():
    partition, reason = _select(NOTHING_IDLE)
    assert partition == "gpu_h100"
    assert "deterministic default" in reason


def test_development_partitions_are_never_selected():
    partition, _ = _select(ONLY_DEV_IDLE)
    assert not partition.startswith("dev_")
    assert partition == "gpu_h100"


def test_alternative_sinfo_format_is_parsed():
    assert _select(ALTERNATIVE_FORMAT)[0] == "gpu_h100_il"


def test_malformed_output_falls_back():
    partition, reason = _select(MALFORMED)
    assert partition == "gpu_h100"
    assert "deterministic default" in reason


def test_unavailable_output_falls_back():
    partition, reason = _select(UNAVAILABLE)
    assert partition == "gpu_h100"
    assert "no idle-node information" in reason


def test_preference_order_is_taken_from_the_argument():
    # A profile may prefer A100; the parser must honour the given order.
    assert _select(H100_IDLE, "gpu_a100_il gpu_h100")[0] == "gpu_a100_il"


def test_explicit_override_validation():
    assert _validate("gpu_h100_il") == 0
    assert _validate("dev_gpu_h100") != 0  # development queue rejected
    assert _validate("gpu_v100") != 0  # not a compatible partition
