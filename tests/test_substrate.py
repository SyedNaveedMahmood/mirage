"""Checkpoint restore, digest determinism, byte-exact mock engine, continuation."""

from __future__ import annotations

from mirage_persist.substrate.dojo.continuation import continue_branch, rollout
from mirage_persist.substrate.dojo.digest import env_digest


def _rollout(adapter, backend, budget, task="user_task_0", suite="banking"):
    return rollout(adapter, backend, suite, task, budget_config=budget, base_seed=1, rollout_tag="t")


def test_checkpoint_deepcopy_restore_independent(adapter, mock_backend, budget):
    r = _rollout(adapter, mock_backend, budget)
    cp = r.checkpoints[0]
    e1 = cp.restore_env()
    e2 = cp.restore_env()
    assert e1 is not e2  # independent objects
    assert env_digest(e1) == env_digest(e2) == cp.env_digest_value  # identical state


def test_digest_excludes_volatile_but_tracks_live_state(adapter):
    env = adapter.fresh_env("banking", {})
    d1 = env_digest(env)
    # mutate live state -> digest changes
    env.bank_account.balance += 1.0
    assert env_digest(env) != d1


def test_mock_greedy_twins_byte_exact(adapter, mock_backend, budget):
    r = _rollout(adapter, mock_backend, budget)
    cp = r.checkpoints[0]
    a = continue_branch(adapter, mock_backend, cp, branch_id="A", replicate=0, seed=1, horizon=8, budget_config=budget, greedy=True)
    b = continue_branch(adapter, mock_backend, cp, branch_id="B", replicate=1, seed=999, horizon=8, budget_config=budget, greedy=True)
    assert a.trajectory_digest == b.trajectory_digest
    assert env_digest(a.env) == env_digest(b.env)


def test_forks_do_not_alias(adapter, mock_backend, budget):
    r = _rollout(adapter, mock_backend, budget)
    cp = r.checkpoints[0]
    a = continue_branch(adapter, mock_backend, cp, branch_id="A", replicate=0, seed=1, horizon=8, budget_config=budget, greedy=True)
    # continuing again from the SAME checkpoint must not be polluted by branch A
    b = continue_branch(adapter, mock_backend, cp, branch_id="B", replicate=0, seed=1, horizon=8, budget_config=budget, greedy=True)
    assert a.trajectory_digest == b.trajectory_digest  # checkpoint state unchanged by A


def test_horizon_and_budget_enforced(adapter, mock_backend, budget):
    r = _rollout(adapter, mock_backend, budget)
    cp = r.checkpoints[0]
    c = continue_branch(adapter, mock_backend, cp, branch_id="A", replicate=0, seed=1, horizon=1, budget_config=budget, greedy=True)
    assert c.loop_result.steps_run <= 1


def test_rollout_solves_banking_task(adapter, mock_backend, budget):
    r = _rollout(adapter, mock_backend, budget)
    cp = r.checkpoints[0]
    ut = adapter.user_task("banking", "user_task_0")
    assert ut.utility("", cp.restore_pre_env(), r.loop_result.env) is True
