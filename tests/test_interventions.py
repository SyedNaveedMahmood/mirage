"""Interventions: >=6 classes, injection changes env, removal is byte-exact + verified."""

from __future__ import annotations

from mirage_persist.interventions import all_interventions, get_intervention, intervention_names
from mirage_persist.substrate.dojo.digest import env_digest


def test_has_six_goal_bearing_classes_plus_controls():
    assert len(intervention_names(exclude_controls=True)) >= 6
    names = set(all_interventions())
    assert "benign_artifact" in names and "negative_control" in names


def test_injection_changes_env_and_text_present(adapter):
    ut = adapter.user_task("banking", "user_task_0")
    it = adapter.injection_task("banking", "injection_task_0")
    iv = get_intervention("injected_instruction")
    inj = iv.build_injections(adapter, "banking", it)
    treated = ut.init_environment(adapter.fresh_env("banking", inj))
    untreated = iv.removal.untreated_env(adapter, "banking", ut)
    assert env_digest(treated) != env_digest(untreated)
    needle = iv.render(it.GOAL).strip()[:40]
    assert needle in treated.model_dump_json()
    assert needle not in untreated.model_dump_json()


def test_removal_postcondition_and_byte_exact(adapter):
    ut = adapter.user_task("banking", "user_task_0")
    it = adapter.injection_task("banking", "injection_task_0")
    iv = get_intervention("poisoned_tool_metadata")  # contains a quote -> exercises YAML escaping
    untreated = iv.removal.untreated_env(adapter, "banking", ut)
    ud = env_digest(untreated)
    # a second untreated build is byte-exact
    assert env_digest(iv.removal.untreated_env(adapter, "banking", ut)) == ud
    # removal postcondition holds
    assert iv.removal.verify_removed(untreated, iv.render(it.GOAL), ud) is True


def test_all_classes_inject_without_breaking_yaml(adapter):
    it = adapter.injection_task("banking", "injection_task_0")
    for name in intervention_names():
        iv = get_intervention(name)
        inj = iv.build_injections(adapter, "banking", it)
        # must not raise (YAML-safe substitution)
        env = adapter.fresh_env("banking", inj)
        assert env is not None
