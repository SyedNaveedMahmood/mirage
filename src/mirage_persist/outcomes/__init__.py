"""Deterministic outcome scoring (no LLM judge for any CV-0 outcome).

    schema.OutcomeVector   -- the componentwise outcome vector (design 2.2)
    scoring.score_continuation -- computes it from a continuation vs. its checkpoint
    action_class           -- classifies each action into the 6 action classes
    subgoals               -- subgoal decomposition -> graded Y^prog + objective graph
"""

from __future__ import annotations

from mirage_persist.outcomes.schema import ACTION_CLASSES, OutcomeVector, outcome_row_names
from mirage_persist.outcomes.scoring import score_continuation
from mirage_persist.outcomes.subgoals import SubgoalSet, get_subgoals, register_subgoals

__all__ = [
    "OutcomeVector",
    "ACTION_CLASSES",
    "outcome_row_names",
    "score_continuation",
    "SubgoalSet",
    "get_subgoals",
    "register_subgoals",
]
