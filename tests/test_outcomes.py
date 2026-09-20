"""Deterministic outcome scoring + subgoal decomposition."""

from __future__ import annotations

from mirage_persist.outcomes.scoring import score_continuation
from mirage_persist.outcomes.subgoals import get_subgoals
from mirage_persist.substrate.dojo.continuation import continue_branch, rollout


def _score_first(adapter, backend, budget, horizons=(1, 3, 5, 10)):
    r = rollout(adapter, backend, "banking", "user_task_0", budget_config=budget, base_seed=1, rollout_tag="t")
    cp = r.checkpoints[0]
    cont = continue_branch(adapter, backend, cp, branch_id="N", replicate=0, seed=7, horizon=10, budget_config=budget, greedy=True)
    return score_continuation(adapter, cp, cont, horizons=list(horizons))


def test_branch_return_and_progress(adapter, mock_backend, budget):
    ov = _score_first(adapter, mock_backend, budget)
    # mock follows ground truth -> returns to GT (Y_branch=0) and solves task (Y_prog=0)
    assert all(v == 0.0 for v in ov.y_branch.values())
    assert ov.y_prog_T == 0.0
    assert ov.y_sec_T == 0.0
    assert ov.invalid_action_rate == 0.0


def test_action_classes_counted(adapter, mock_backend, budget):
    ov = _score_first(adapter, mock_backend, budget)
    assert ov.action_class_counts["user_read"] >= 1
    assert ov.action_class_counts["user_write"] >= 1
    assert ov.action_class_counts["terminate"] >= 1


def test_outcome_row_has_horizon_columns(adapter, mock_backend, budget):
    ov = _score_first(adapter, mock_backend, budget)
    row = ov.to_row()
    for h in (1, 3, 5, 10):
        assert f"Y_branch_{h}" in row
    assert "Y_prog_T" in row and "Y_sec_T" in row


def test_subgoals_graded_and_bounded(adapter):
    ut = adapter.user_task("banking", "user_task_0")
    sg = get_subgoals("banking", "user_task_0", ut)
    assert not sg.is_fallback  # banking user_task_0 has a hand-written decomposition
    assert len(sg.subgoals) == 2
    # y_prog is in [0,1]
    pre = adapter.fresh_env("banking", {})
    y = sg.y_prog("", pre, pre, [])
    assert 0.0 <= y <= 1.0


def test_fallback_subgoals_for_unregistered_task(adapter):
    ut = adapter.user_task("banking", "user_task_1")
    sg = get_subgoals("banking", "user_task_1", ut)
    assert sg.is_fallback
    assert len(sg.subgoals) == 1
