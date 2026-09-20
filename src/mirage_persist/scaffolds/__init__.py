"""Agent scaffolds. S1 (ReAct, clean control) and S2 (planner-executor with notes).

For CV-0 a scaffold configures the system message (a real behavioural difference for
LLM backends) and seeds ``scaffold_state`` (the note store carried in checkpoints).
The full note-store dynamics + the memory (M) reset land in CV-2; the plumbing
(state field, snapshot in the checkpoint) exists here so CV-2 slots in.
"""

from __future__ import annotations

from mirage_persist.scaffolds.base import Scaffold, build_scaffold

__all__ = ["Scaffold", "build_scaffold"]
