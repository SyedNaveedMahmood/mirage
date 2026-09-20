"""Offline Qwen size-comparison contracts and mocked Slurm submission tests.

The fake sbatch records requests; it never runs a job body or contacts Slurm.
"""

from __future__ import annotations

import csv
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from mirage_persist.config.loader import load_config
from shell_support import CLUSTER_DIR, REPO_ROOT, find_bash

BASH = find_bash()
SIZES = ("4b", "9b", "27b")
SHELL = pytest.mark.skipif(BASH is None, reason="POSIX bash required")


def test_size_configs_are_identical_except_model_and_labels():
    configs = [load_config(REPO_ROOT / f"configs/cv0/cv0_qwen3p5_{s}_100cp.yaml") for s in SIZES]
    comparable = [c.model_dump(exclude={"models", "experiment_id", "notes"}) for c in configs]
    assert comparable[0] == comparable[1] == comparable[2]
    assert len({c.experiment_id for c in configs}) == 3
    for size, config in zip(SIZES, configs, strict=True):
        assert config.determinism_audit.n_checkpoints == 100
        assert config.determinism_audit.K == 16
        assert len(config.models) == 1
        model = config.models[0]
        assert model.model_id == f"Qwen/Qwen3.5-{size.upper()}"
        assert model.family == "qwen3.5"
        assert model.reasoning_mode == "non-thinking"
        assert model.resolved_extra_body()["chat_template_kwargs"] == {"enable_thinking": False}
    sampling = [c.models[0].sampling.model_dump() for c in configs]
    assert sampling[0] == sampling[1] == sampling[2]


def _clean_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("SBATCH_", "MIRAGE_", "VLLM_", "MOCK_")) or key in {
            "CV0_CONFIG", "SKIP_MODEL_PREFETCH", "HF_HOME", "AGENTRUN_ROOT", "PYTHON_BIN",
        }:
            env.pop(key)
    env["PYTHON_BIN"] = sys.executable.replace("\\", "/")
    return env


@SHELL
def test_off_cluster_dry_run_is_read_only(tmp_path):
    # An arbitrary layout, with no Git repo, Slurm or vLLM environment.
    checkout = tmp_path / "arbitrary"
    shutil.copytree(CLUSTER_DIR, checkout / "cluster/bwunicluster3")
    shutil.copytree(REPO_ROOT / "configs", checkout / "configs")
    before = sorted(str(p.relative_to(checkout)) for p in checkout.rglob("*"))
    result = subprocess.run(
        [BASH, (checkout / "cluster/bwunicluster3/submit_cv0_qwen35_sequence.sh").as_posix(), "--dry-run"],
        env=_clean_env(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "afterany" in result.stdout
    assert all(f"Qwen3.5-{s.upper()}" in result.stdout for s in SIZES)
    assert "no jobs" in result.stdout.lower()
    assert before == sorted(str(p.relative_to(checkout)) for p in checkout.rglob("*"))


def _executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def cluster(tmp_path):
    if BASH is None:
        pytest.skip("POSIX bash required")
    root = tmp_path / "agentrun" / "mirage-persist"
    scripts = root / "cluster/bwunicluster3"
    shutil.copytree(CLUSTER_DIR, scripts)
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    python = shlex.quote(sys.executable.replace("\\", "/"))
    _executable(root / ".venv/bin/python", f'#!/usr/bin/env bash\nexec {python} "$@"\n')
    _executable(root / ".venv/bin/mirage", "#!/usr/bin/env bash\nexit 99\n")
    _executable(root.parent / "vllm/.venv/bin/vllm", "#!/usr/bin/env bash\nexit 99\n")
    fake_bin = tmp_path / "fake-bin"
    _executable(fake_bin / "sbatch", """#!/usr/bin/env bash
set -Eeuo pipefail
n=100
if [[ -f "${MOCK_COUNT}" ]]; then read -r n < "${MOCK_COUNT}"; fi
n=$((n + 1))
printf '%s\\n' "$n" > "${MOCK_COUNT}"
printf '%s\\t' "$@" >> "${MOCK_LOG}"
printf '\\n' >> "${MOCK_LOG}"
if [[ "$n" == "${MOCK_FAIL_AT:-}" ]]; then echo 'mock submission rejected' >&2; exit 1; fi
if [[ "$n" == "${MOCK_BAD_ID_AT:-}" ]]; then echo 'invalid-id'; exit 0; fi
printf '%s;testcluster\\n' "$n"
""")
    env = _clean_env()
    env.update({
        "MOCK_LOG": (tmp_path / "sbatch.tsv").as_posix(),
        "MOCK_COUNT": (tmp_path / "count").as_posix(),
        "PATH": str(fake_bin) + os.pathsep + env["PATH"],
    })
    return root, scripts, env


def _run(cluster, *args, script="submit_cv0_qwen35_sequence.sh"):
    root, scripts, env = cluster
    return subprocess.run([BASH, (scripts / script).as_posix(), *args],
                          cwd=root, env=env, capture_output=True, text=True)


def _calls(cluster):
    from pathlib import Path

    log = Path(cluster[2]["MOCK_LOG"])
    return [line.rstrip("\t").split("\t") for line in log.read_text().splitlines()] if log.exists() else []


def _ledgers(cluster):
    rows = []
    for path in (cluster[0] / "runs/_cluster_campaigns").glob("*.tsv"):
        with path.open() as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    return rows


@pytest.mark.parametrize("policy,options", [("afterany", []), ("afterok", ["--stop-on-failure"])])
def test_jobs_are_chained_with_prefetch_dependencies(cluster, policy, options):
    result = _run(cluster, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(cluster)
    assert len(calls) == 6
    for index, size in enumerate(SIZES):
        prefetch, gpu = calls[index * 2:index * 2 + 2]
        assert f"--job-name=fetch-qwen3p5_{size}" in prefetch
        assert f"--job-name=cv0-qwen3p5_{size}" in gpu
        assert not any(a.startswith("--dependency") for a in prefetch)
        dependencies = f"afterok:{101 + index * 2}"
        if index:
            dependencies = f"{policy}:{100 + index * 2}," + dependencies
        assert f"--dependency={dependencies}" in gpu
        assert "--kill-on-invalid-dep=yes" in gpu
        assert any("CV0_CONFIG=" in a and f"_{size}_100cp.yaml" in a for a in gpu)
        if size == "27b":
            assert "--partition=gpu_h100,gpu_h100_il" in gpu
    ledger = _ledgers(cluster)
    assert len(ledger) == 6
    assert [r["job_id"] for r in ledger if r["kind"] == "gpu"] == ["102", "104", "106"]
    assert "not yet completed" in result.stdout


@pytest.mark.parametrize("mode", ["skip", "cached"])
def test_no_prefetch_still_orders_gpus(cluster, mode):
    root, _, env = cluster
    if mode == "skip":
        env["SKIP_MODEL_PREFETCH"] = "1"
    else:
        markers = root.parent / "vllm/cache/huggingface/.mirage_prefetch"
        markers.mkdir(parents=True)
        for size in SIZES:
            config = load_config(root / f"configs/cv0/cv0_qwen3p5_{size}_100cp.yaml")
            (markers / f"qwen3p5_{size}@{config.models[0].revision}.done").touch()
    result = _run(cluster)
    assert result.returncode == 0, result.stderr
    calls = _calls(cluster)
    assert len(calls) == 3
    assert not any(a.startswith("--dependency=") for a in calls[0])
    assert "--dependency=afterany:101" in calls[1]
    assert "--dependency=afterany:102" in calls[2]


@pytest.mark.parametrize("override", [
    {"CV0_CONFIG": "anything.yaml"},
    {"SBATCH_GPU_PARTITION": "dev_gpu_h100"},
    {"SBATCH_GPU_PARTITION": "gpu_a100_il"},  # rejected by the third profile, before any job
    {"SBATCH_GPU_TIME": "48:00:00"},
])
def test_invalid_campaign_environment_submits_nothing(cluster, override):
    cluster[2].update(override)
    assert _run(cluster).returncode != 0
    assert _calls(cluster) == []


def test_profile_config_mismatch_submits_nothing(cluster):
    config = cluster[0] / "configs/models/qwen3p5_27b.yaml"
    config.write_text(config.read_text().replace("fc05daec18b0a78c049392ed2e771dde82bdf654", "0" * 40))
    result = _run(cluster)
    assert result.returncode != 0
    assert "profile/config mismatch" in result.stderr
    assert _calls(cluster) == []


@pytest.mark.parametrize("original,replacement", [
    ("base_url: null", 'base_url: "http://example.invalid/v1"'),
    ('base_url_env: "OPENAI_COMPATIBLE_BASE_URL"', 'base_url_env: "OTHER_URL"'),
    ('api_key_env: "OPENAI_COMPATIBLE_API_KEY"', 'api_key_env: "OTHER_KEY"'),
])
def test_cluster_config_cannot_bypass_allocation_server(cluster, original, replacement):
    config = cluster[0] / "configs/models/qwen3p5_27b.yaml"
    config.write_text(config.read_text().replace(original, replacement))
    result = _run(cluster)
    assert result.returncode != 0
    assert "allocation's" in result.stderr
    assert _calls(cluster) == []


@pytest.mark.parametrize("failure", ["MOCK_FAIL_AT", "MOCK_BAD_ID_AT"])
def test_partial_submission_stops_and_preserves_accepted_ids(cluster, failure):
    cluster[2][failure] = "104"
    result = _run(cluster)
    assert result.returncode != 0
    assert len(_calls(cluster)) == 4  # no third model submitted
    assert [r["job_id"] for r in _ledgers(cluster)] == ["101", "102", "103"]
    assert "NOT rolled back" in result.stderr
    assert "scancel" in result.stderr


def test_parsable_wrapper_only_returns_gpu_id(cluster):
    result = _run(cluster, "qwen3p5_4b", "--parsable", script="submit_cv0_model.sh")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "102\n"
    assert "Config validated" in result.stderr


@pytest.mark.parametrize("args", [["--unknown"], ["qwen3p5_4b", "--dependency", "afterany:1,bad"]])
def test_unknown_or_malformed_options_submit_nothing(cluster, args):
    script = "submit_cv0_model.sh" if args[0] == "qwen3p5_4b" else "submit_cv0_qwen35_sequence.sh"
    assert _run(cluster, *args, script=script).returncode != 0
    assert _calls(cluster) == []
