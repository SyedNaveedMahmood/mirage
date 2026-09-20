"""Carrier reset operators (design 4.5).

The eight carriers C, M, E, B, H, U, P, L are registered here with the reset
interface and the paired *untreated-reset twin* API (every reset takes the
never-treated reference whose values it restores). CV-0 does not reset outcomes;
this registry exists so CV-2 slots in without refactor. The memory (M) reset is
specified as snapshot-restore, never a scrub (Zombie Agents). Operators that need
the full continuation machinery (C context rebuild, L span redaction, P process
reload) are registered as CV-2 and raise if invoked early -- but they are visible,
named, and typed now.
"""

from __future__ import annotations

from mirage_persist.resets.base import BranchState, Carrier, ResetOperator
from mirage_persist.resets.registry import carrier_metadata, get_reset, list_carriers

__all__ = [
    "Carrier",
    "ResetOperator",
    "BranchState",
    "get_reset",
    "list_carriers",
    "carrier_metadata",
]
