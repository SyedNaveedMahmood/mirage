"""DojoAdapter: the single boundary between MIRAGE and AgentDojo.

Everything MIRAGE needs from AgentDojo goes through here: loading suites/tasks,
building untreated vs. treated (intervention-injected) environments, constructing
the per-task tool runtime and the ToolsExecutor, and the default system message.
Keeping this in one place means a substrate upgrade touches one file.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, TaskEnvironment
from agentdojo.task_suite.load_suites import get_suite as _ad_get_suite
from agentdojo.task_suite.task_suite import TaskSuite, functions_stack_trace_from_messages
from agentdojo.types import ChatMessage, ChatSystemMessage, ChatUserMessage, text_content_block_from_string


@lru_cache(maxsize=32)
def _cached_suite(benchmark_version: str, suite_name: str) -> TaskSuite:
    return _ad_get_suite(benchmark_version, suite_name)


class DojoAdapter:
    """Thin, cached accessor for one AgentDojo benchmark version."""

    def __init__(self, benchmark_version: str = "v1.2.2") -> None:
        self.benchmark_version = benchmark_version

    # -- suites / tasks --------------------------------------------------- #
    def suite(self, suite_name: str) -> TaskSuite:
        return _cached_suite(self.benchmark_version, suite_name)

    def user_task_ids(self, suite_name: str) -> list[str]:
        return list(self.suite(suite_name).user_tasks.keys())

    def injection_task_ids(self, suite_name: str) -> list[str]:
        return list(self.suite(suite_name).injection_tasks.keys())

    def user_task(self, suite_name: str, task_id: str) -> BaseUserTask:
        return self.suite(suite_name).get_user_task_by_id(task_id)

    def injection_task(self, suite_name: str, task_id: str) -> BaseInjectionTask:
        return self.suite(suite_name).get_injection_task_by_id(task_id)

    def injection_vectors(self, suite_name: str) -> dict[str, str]:
        return self.suite(suite_name).get_injection_vector_defaults()

    # -- environments ----------------------------------------------------- #
    def fresh_env(self, suite_name: str, injections: dict[str, str] | None = None) -> TaskEnvironment:
        """A freshly loaded environment. ``injections={}`` (default) is the untreated baseline."""
        return self.suite(suite_name).load_and_inject_default_environment(injections or {})

    def init_env_for_task(
        self, suite_name: str, user_task: BaseUserTask, injections: dict[str, str] | None = None
    ) -> tuple[TaskEnvironment, TaskEnvironment]:
        """Return ``(task_env, pre_env)``: the task-initialized env and its deep-copied
        scoring baseline (mirrors AgentDojo's ``run_task_with_pipeline`` setup)."""
        env = self.fresh_env(suite_name, injections)
        task_env = user_task.init_environment(env)
        pre_env = task_env.model_copy(deep=True)
        return task_env, pre_env

    # -- runtime / executor / prompt ------------------------------------- #
    def runtime(self, suite_name: str) -> FunctionsRuntime:
        return FunctionsRuntime(self.suite(suite_name).tools)

    @staticmethod
    def tools_executor() -> ToolsExecutor:
        return ToolsExecutor()

    @staticmethod
    def system_message(name: str | None = None) -> str:
        return load_system_message(name)

    def initial_messages(
        self, user_task: BaseUserTask | BaseInjectionTask, system_message: str | None = None
    ) -> list[ChatMessage]:
        """Build the [system, user] prefix that seeds a rollout (scaffold may override the system message)."""
        prompt = self.prompt_for(user_task)
        sys_text = system_message if system_message is not None else self.system_message()
        system = ChatSystemMessage(role="system", content=[text_content_block_from_string(sys_text)])
        user = ChatUserMessage(role="user", content=[text_content_block_from_string(prompt)])
        return [system, user]

    @staticmethod
    def prompt_for(task: BaseUserTask | BaseInjectionTask) -> str:
        return task.PROMPT if isinstance(task, BaseUserTask) else task.GOAL

    @staticmethod
    def ground_truth_calls(task: BaseUserTask | BaseInjectionTask, env: TaskEnvironment) -> list[FunctionCall]:
        return list(task.ground_truth(env))

    @staticmethod
    def trace_from_messages(messages: Sequence[ChatMessage]) -> list[FunctionCall]:
        return functions_stack_trace_from_messages(messages)


def check_suite(benchmark_version: str, suite_name: str, *, check_injectable: bool = True) -> bool:
    """Run AgentDojo's own suite self-check; print a summary; return overall pass."""
    suite = _cached_suite(benchmark_version, suite_name)
    ok, (user_results, injection_results) = suite.check(check_injectable=check_injectable)
    n_user_ok = sum(1 for v in user_results.values() if v[0])
    n_inj_ok = sum(1 for v in injection_results.values() if v)
    print(f"[check-substrate] {benchmark_version}/{suite_name}: overall={'PASS' if ok else 'FAIL'}")
    print(f"[check-substrate] user tasks OK: {n_user_ok}/{len(user_results)}")
    print(f"[check-substrate] injection tasks OK: {n_inj_ok}/{len(injection_results)}")
    for tid, (passed, reason) in user_results.items():
        if not passed:
            print(f"[check-substrate][fail] {tid}: {reason}")
    return bool(ok)


__all__ = ["DojoAdapter", "check_suite"]
