"""End-to-end CV-0 (mock): all four sub-experiments run, gates fire, artifacts exist."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirage_persist.config.loader import load_config
from mirage_persist.config.schema import CV0Config
from mirage_persist.experiments.cv0.report import run_cv0
from mirage_persist.run.run_context import RunContext

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_cv0_mock_smoke_passes(tmp_path):
    cfg: CV0Config = load_config(
        _ROOT / "configs" / "smoke" / "cv0_smoke_mock.yaml",
        CV0Config,
        overrides={"determinism_audit.n_checkpoints": "4", "capability_screen.max_tasks_per_suite": "3"},
    )
    cfg.paths.runs_dir = tmp_path / "runs"
    cfg.paths.artifacts_dir = tmp_path / "artifacts"

    with RunContext(cfg, subcommand="test smoke") as ctx:
        report = run_cv0(cfg, ctx)

    # mock is byte-exact -> determinism gate PASS and overall PASS
    assert report.overall == "PASS"
    gate_status = {g["name"]: g["status"] for g in report.gates}
    assert gate_status["digest_equality"] == "PASS"
    assert gate_status["twin_null_branch"] == "PASS"

    # every canonical artifact was written
    for name in (
        "envelope.json",
        "eligibility.parquet",
        "capability_report.md",
        "figure_s1_twin_split_null.png",
        "potency.json",
        "variance_power.json",
        "cv0_verdict.json",
        "manifest.json",
        "terminal.log",
        "events.jsonl",
    ):
        assert (ctx.run_dir / name).exists(), f"missing artifact: {name}"

    # envelope epsilon floor for the branch outcome is ~0 for the mock engine
    import json

    env = json.loads((ctx.run_dir / "envelope.json").read_text(encoding="utf-8"))
    assert env["digest_equality_rate"] == 1.0

    # the event stream is emitted from ALL screens (not just the determinism audit)
    roles = set()
    for line in (ctx.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        roles.add(json.loads(line)["role"])
    assert {"score", "eligibility", "potency"} <= roles
