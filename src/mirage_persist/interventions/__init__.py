"""Versioned interventions + removal operators (design 4.4, intervention taxonomy).

An ``Intervention`` renders text into AgentDojo injection vectors (data-level, via
``str.format`` before parse). A ``RemovalOperator`` restores the untreated env
byte-for-byte and exposes a machine-checkable postcondition. CV-0's potency screen
uses exposure only; the removal operators are built and unit-tested here so CV-1
reuses them unchanged.
"""

from __future__ import annotations

from mirage_persist.interventions.base import Intervention, RemovalOperator
from mirage_persist.interventions.registry import (
    all_interventions,
    get_intervention,
    intervention_names,
)

__all__ = [
    "Intervention",
    "RemovalOperator",
    "get_intervention",
    "all_interventions",
    "intervention_names",
]
