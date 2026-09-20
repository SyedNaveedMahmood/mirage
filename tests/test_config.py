"""Config loader: extends, ref-inlining, overrides, strictness, hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mirage_persist.config import canonical_config_hash, load_config
from mirage_persist.config.schema import CV0Config

_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_mock_config_loads():
    cfg = load_config(_ROOT / "configs" / "smoke" / "cv0_smoke_mock.yaml", CV0Config)
    assert cfg.experiment_id == "cv0_smoke_mock"
    assert cfg.substrate.suites == ["banking", "slack"]
    assert cfg.models[0].backend.value == "mock"
    assert {s.name for s in cfg.scaffolds} == {"S1", "S2"}


def test_extends_and_override():
    cfg = load_config(
        _ROOT / "configs" / "cv0" / "cv0_full.yaml",
        CV0Config,
        overrides={"determinism_audit.n_checkpoints": "7"},
    )
    assert cfg.determinism_audit.n_checkpoints == 7
    # inherited from base.yaml
    assert cfg.substrate.suites == ["banking", "slack", "travel", "workspace"]


def test_bwunicluster_qwen_config_loads_full_suite():
    cfg = load_config(
        _ROOT / "configs" / "cv0" / "cv0_bwunicluster_qwen3p5_9b.yaml",
        CV0Config,
    )
    assert cfg.experiment_id == "cv0_bwunicluster_qwen3p5_9b"
    assert cfg.determinism_audit.n_checkpoints == 100  # inherited from cv0_full.yaml
    assert cfg.determinism_audit.K == 16
    assert cfg.substrate.suites == ["banking", "slack", "travel", "workspace"]
    assert {s.name for s in cfg.scaffolds} == {"S1", "S2"}
    assert len(cfg.models) == 1
    model = cfg.models[0]
    assert model.model_id == "Qwen/Qwen3.5-9B"
    assert model.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert model.reasoning_mode == "non-thinking"
    assert cfg.gates.min_families_pass_capability == 1
    assert cfg.gates.kill_min_families == 1


def test_strictness_rejects_typo():
    with pytest.raises(ValidationError):
        load_config(
            _ROOT / "configs" / "smoke" / "cv0_smoke_mock.yaml",
            CV0Config,
            overrides={"determinism_audit.n_checkpointz": "10"},
        )


def test_config_hash_stable_and_path_independent():
    cfg1 = load_config(_ROOT / "configs" / "smoke" / "cv0_smoke_mock.yaml", CV0Config)
    cfg2 = load_config(_ROOT / "configs" / "smoke" / "cv0_smoke_mock.yaml", CV0Config)
    assert canonical_config_hash(cfg1) == canonical_config_hash(cfg2)
    # changing an output path must NOT change the scientific hash
    cfg2.paths.runs_dir = Path("some/other/dir")
    assert canonical_config_hash(cfg1) == canonical_config_hash(cfg2)


def test_shared_single_family_profile_is_not_runnable():
    with pytest.raises(ValidationError):
        load_config(
            _ROOT / "configs" / "cv0" / "cv0_cluster_single_family_100cp.yaml",
            CV0Config,
        )
