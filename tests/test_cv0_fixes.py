"""Regression tests for the three CV-0 validity fixes.

1a. wall-clock timestamps are excluded from the reproducibility digest (travel/workspace
    email + cloud-drive tools stamp datetime.now(), which used to break twin equality);
1b. determinism-audit checkpoint collection is stratified across suites and scaffolds;
2.  GPU-hours are recorded for the server (openai-compatible) topology;
3.  the event stream is emitted from every screen, not just the determinism audit.
"""

from __future__ import annotations

from mirage_persist.config.schema import (
    BackendType,
    CV0Config,
    ModelBackendConfig,
    ScaffoldConfig,
    ScaffoldType,
)
from mirage_persist.experiments.cv0.common import collect_checkpoints
from mirage_persist.models.base import build_backend
from mirage_persist.run.run_context import RunContext
from mirage_persist.scaffolds.base import build_scaffold
from mirage_persist.substrate.dojo.continuation import continue_branch, rollout
from mirage_persist.substrate.dojo.digest import DEFAULT_VOLATILE_KEYS, env_digest


# ---- Fix 1a: timestamps excluded from the digest ------------------------- #
def test_volatile_keys_include_wallclock_timestamps():
    assert "timestamp" in DEFAULT_VOLATILE_KEYS
    assert "last_modified" in DEFAULT_VOLATILE_KEYS


def test_digest_ignores_timestamp_fields():
    # two structures differing ONLY in wall-clock fields must hash identically
    from mirage_persist.substrate.dojo.digest import _canonical_json, _strip_volatile

    a = {"emails": {"1": {"body": "hi", "timestamp": "2026-07-23T03:11:09.838804"}}}
    b = {"emails": {"1": {"body": "hi", "timestamp": "2026-07-23T03:11:09.999999"}}}
    assert _canonical_json(_strip_volatile(a, DEFAULT_VOLATILE_KEYS)) == _canonical_json(
        _strip_volatile(b, DEFAULT_VOLATILE_KEYS)
    )


def test_travel_engine_byte_exact_after_timestamp_fix(adapter, mock_backend, budget):
    # travel/user_task_1 sends an email (datetime.now timestamp): used to diverge.
    r = rollout(adapter, mock_backend, "travel", "user_task_1", budget_config=budget, base_seed=1, rollout_tag="t", greedy=True)
    assert r.checkpoints, "expected at least one checkpoint"
    for cp in r.checkpoints:
        a = continue_branch(adapter, mock_backend, cp, branch_id="A", replicate=0, seed=1, horizon=10, budget_config=budget, greedy=True)
        b = continue_branch(adapter, mock_backend, cp, branch_id="B", replicate=1, seed=2, horizon=10, budget_config=budget, greedy=True)
        assert a.trajectory_digest == b.trajectory_digest
        assert env_digest(a.env) == env_digest(b.env)


# ---- Fix 1b: stratified checkpoint collection ---------------------------- #
def _mock_cfg():
    return ModelBackendConfig(name="mock", backend=BackendType.MOCK, model_id="m")


def test_collect_checkpoints_is_stratified(adapter):
    cfg = CV0Config(
        experiment_id="strat",
        models=[_mock_cfg()],
        scaffolds=[
            ScaffoldConfig(name="S1", kind=ScaffoldType.S1_REACT),
            ScaffoldConfig(name="S2", kind=ScaffoldType.S2_PLANNER, notes_enabled=True),
        ],
    )  # default substrate.suites = all four
    backends = [(_mock_cfg(), build_backend(_mock_cfg()))]
    scaffolds = [build_scaffold(s) for s in cfg.scaffolds]
    recs = collect_checkpoints(adapter, cfg, backends, scaffolds, greedy=True, max_checkpoints_total=24)
    assert len(recs) == 24
    # the first 24 checkpoints must span multiple suites AND both scaffolds
    assert len({r.suite for r in recs}) >= 3
    assert {r.scaffold.name for r in recs} == {"S1", "S2"}


# ---- Fix 2: server-topology GPU-hours ------------------------------------ #
def _cfg_with(backend: BackendType) -> CV0Config:
    return CV0Config(
        experiment_id="gpuhrs",
        models=[ModelBackendConfig(name="M", backend=backend, model_id="x")],
        scaffolds=[ScaffoldConfig(name="S1", kind=ScaffoldType.S1_REACT)],
    )


def test_server_backend_detection():
    assert RunContext(_cfg_with(BackendType.OPENAI_COMPATIBLE))._uses_server_backend() is True
    assert RunContext(_cfg_with(BackendType.MOCK))._uses_server_backend() is False


def test_gpu_hours_server_proxy_when_gpu_visible(tmp_path):
    from mirage_persist.run.gpu import list_gpus

    cfg = _cfg_with(BackendType.OPENAI_COMPATIBLE)
    cfg.paths.runs_dir = tmp_path / "runs"
    with RunContext(cfg, subcommand="t") as ctx:
        pass
    gpu = ctx.manifest.gpu
    if list_gpus():  # only assert the proxy where a GPU is actually visible
        assert gpu["gpu_hours"] > 0.0
        assert gpu["gpu_hours_basis"] == "server_side_wall_clock"
    else:
        assert gpu["gpu_hours"] == 0.0
