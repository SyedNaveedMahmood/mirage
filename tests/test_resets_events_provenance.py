"""Resets registry, event JSONL round-trip, and run provenance completeness."""

from __future__ import annotations

import json

import pytest

from mirage_persist.config.loader import load_config
from mirage_persist.config.schema import CV0Config
from mirage_persist.events.schema import Event
from mirage_persist.events.writer import EventWriter
from mirage_persist.resets import get_reset, list_carriers
from mirage_persist.resets.base import BranchState, Carrier
from mirage_persist.run.run_context import RunContext


# ----------------------------- resets ------------------------------------- #
def test_all_eight_carriers_registered():
    assert {c.value for c in list_carriers()} == {"C", "M", "E", "B", "H", "U", "P", "L"}


def test_environment_reset_restores_untreated(adapter):
    treated = adapter.fresh_env("banking", {})
    untreated = adapter.fresh_env("banking", {})
    treated.bank_account.balance += 500.0
    state = BranchState(env=treated, messages=[])
    ref = BranchState(env=untreated, messages=[])
    from mirage_persist.substrate.dojo.digest import env_digest

    get_reset(Carrier.E).reset(state, ref)
    assert env_digest(state.env) == env_digest(untreated)


def test_memory_reset_snapshot_restore():
    state = BranchState(env=None, messages=[], scaffold_state={"notes": ["poisoned"], "plan": "bad"})  # type: ignore[arg-type]
    ref = BranchState(env=None, messages=[], scaffold_state={"notes": [], "plan": ""})  # type: ignore[arg-type]
    # env is unused by the M reset
    get_reset(Carrier.M).reset(state, ref)
    assert state.scaffold_state["notes"] == []
    assert state.scaffold_state["plan"] == ""


def test_cv2_placeholder_reset_raises():
    with pytest.raises(NotImplementedError):
        get_reset("C").reset(None, None)  # type: ignore[arg-type]


# ----------------------------- events ------------------------------------- #
def test_event_jsonl_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventWriter(path) as w:
        w.emit(checkpoint_id="c1", branch_id="N", step=0, role="score", seed=42, env_digest="abc")
        w.emit(checkpoint_id="c1", branch_id="N", step=1, role="decision")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ev = Event(**json.loads(lines[0]))
    assert ev.checkpoint_id == "c1" and ev.seed == 42 and ev.env_digest == "abc"


# --------------------------- provenance ----------------------------------- #
def test_run_manifest_has_required_fields(tmp_path):
    from pathlib import Path

    cfg: CV0Config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "smoke" / "cv0_smoke_mock.yaml", CV0Config
    )
    cfg.paths.runs_dir = tmp_path / "runs"
    with RunContext(cfg, subcommand="test") as ctx:
        pass
    manifest = json.loads((ctx.run_dir / "manifest.json").read_text(encoding="utf-8"))
    for field in (
        "run_id", "experiment_id", "started_at", "ended_at", "duration_s", "hostname",
        "python_version", "gpu", "mirage_git", "substrate_git", "substrate_expected_sha",
        "dep_versions", "config_hash", "global_seed", "terminal_log_path", "env_freeze_sha256",
    ):
        assert field in manifest, f"missing provenance field: {field}"
    assert manifest["dep_versions"]["agentdojo"] == "0.1.35"
    assert (ctx.run_dir / "terminal.log").exists()
    assert (ctx.run_dir / "resolved_config.yaml").exists()
