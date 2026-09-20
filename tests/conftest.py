"""Shared pytest fixtures."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from mirage_persist.config.schema import BackendType, BudgetConfig, ModelBackendConfig, ScaffoldConfig, ScaffoldType
from mirage_persist.models.base import build_backend
from mirage_persist.scaffolds.base import build_scaffold
from mirage_persist.substrate.dojo.adapter import DojoAdapter

BENCHMARK_VERSION = "v1.2.2"


@pytest.fixture(scope="session")
def adapter() -> DojoAdapter:
    return DojoAdapter(BENCHMARK_VERSION)


@pytest.fixture()
def mock_backend():
    return build_backend(ModelBackendConfig(name="mock", backend=BackendType.MOCK, model_id="mock-v1"))


@pytest.fixture()
def budget() -> BudgetConfig:
    return BudgetConfig(max_steps=12)


@pytest.fixture()
def s1():
    return build_scaffold(ScaffoldConfig(name="S1", kind=ScaffoldType.S1_REACT))
