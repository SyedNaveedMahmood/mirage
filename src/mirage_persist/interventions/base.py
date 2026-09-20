"""Intervention + RemovalOperator base types.

Data-level injection: an intervention renders text and places it into one or more
AgentDojo injection vectors; the environment is then re-instantiated with those
substitutions. Because tool outputs are a pure function of env state, *removal* is
structural -- rebuild the env with the untreated (default) vector values -- and its
postcondition is machine-checkable (untreated digest match + injected text absent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import TaskEnvironment

from mirage_persist.substrate.dojo.adapter import DojoAdapter
from mirage_persist.substrate.dojo.digest import env_digest


def _yaml_dq_escape(text: str) -> str:
    """Escape a value so it is safe inside a double-quoted YAML scalar.

    AgentDojo injects via ``environment_text.format(**injections)`` then
    ``yaml.safe_load``; every injection placeholder lives inside a double-quoted
    scalar (verified across all suites). YAML un-escapes this at parse time, so the
    agent sees the original text -- the escaping only keeps the YAML valid.
    """
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "")
    )


@dataclass
class RemovalOperator:
    """Restore the untreated environment and verify the intervention is gone."""

    def untreated_env(self, adapter: DojoAdapter, suite_name: str, user_task: BaseUserTask) -> TaskEnvironment:
        env = adapter.fresh_env(suite_name, injections={})
        return user_task.init_environment(env)

    @staticmethod
    def verify_removed(env: TaskEnvironment, injected_text: str, untreated_digest: str) -> bool:
        """Postcondition: env digest equals the untreated baseline AND the injected
        text is absent from the serialized environment (no residual in any field)."""
        if env_digest(env) != untreated_digest:
            return False
        needle = injected_text.strip()
        if not needle:
            return True
        blob = env.model_dump_json()
        return needle not in blob


@dataclass
class Intervention:
    """A versioned, class-tagged intervention template + its removal operator."""

    name: str
    version: str
    intervention_class: str  # injected_instruction | misleading_factual | poisoned_tool_metadata |
    #                          false_relationship | salient_irrelevant | adjacent_relevant | benign_artifact | negative_control
    relevance: str  # task-relevant | adjacent | irrelevant-salient | none
    template: str  # may contain {goal}; braces in the SUBSTITUTED value are safe (not re-formatted)
    goal_bearing: bool = True  # False for BA / salient_irrelevant / NC
    removal: RemovalOperator = field(default_factory=RemovalOperator)

    def render(self, goal: str = "") -> str:
        if not self.goal_bearing:
            return self.template
        return self.template.replace("{goal}", goal)

    def build_injections(
        self,
        adapter: DojoAdapter,
        suite_name: str,
        injection_task: BaseInjectionTask | None,
        *,
        vectors: list[str] | None = None,
    ) -> dict[str, str]:
        """Return the {vector_id: text} map to hand to ``load_and_inject_default_environment``."""
        goal = injection_task.GOAL if (injection_task is not None and self.goal_bearing) else ""
        text = self.render(goal)
        target_vectors = vectors if vectors is not None else list(adapter.injection_vectors(suite_name).keys())
        if self.intervention_class == "negative_control":
            # NC mutates env state but leaves no agent-visible *instructional* evidence:
            # substitute a whitespace-only value (present in state, conveys nothing).
            return {v: " " for v in target_vectors}
        escaped = _yaml_dq_escape(text)
        return {v: escaped for v in target_vectors}


__all__ = ["Intervention", "RemovalOperator"]
