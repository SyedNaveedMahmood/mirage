"""Substrate A -- `MIRAGE-Dojo`: exact checkpoint/fork/continue over AgentDojo.

Public surface:

    adapter.DojoAdapter          -- load suites/tasks/envs, build tool runtimes
    digest.env_digest / trajectory_digest
    checkpoint.Checkpoint         -- an immutable, deep-copied pre-decision state
    agent_loop.run_agent_loop     -- ReAct loop with a pre-decision checkpoint hook
    continuation.fork / continue_branch
    budget.BudgetMeter
"""

from __future__ import annotations

from mirage_persist.substrate.dojo.adapter import DojoAdapter, check_suite
from mirage_persist.substrate.dojo.budget import BudgetMeter, BudgetSnapshot
from mirage_persist.substrate.dojo.checkpoint import Checkpoint
from mirage_persist.substrate.dojo.digest import env_digest, trajectory_digest

__all__ = [
    "DojoAdapter",
    "check_suite",
    "BudgetMeter",
    "BudgetSnapshot",
    "Checkpoint",
    "env_digest",
    "trajectory_digest",
]
