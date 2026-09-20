"""Carrier reset registry: the eight carriers, their metadata, and operators.

CV-0 implements the cleanly-implementable carriers (E environment, M memory-notes)
and registers the rest (C, B, H, U, P, L) as CV-2 placeholders so the full carrier
set is visible and typed now. The untreated-reset twin is the ``untreated`` argument
to every operator.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage_persist.resets.base import BranchState, Carrier, CV2Placeholder, ResetOperator


@dataclass
class CarrierMeta:
    carrier: Carrier
    contents: str
    reset_description: str
    implemented_in: str  # "CV-0" or "CV-2"


class EnvironmentReset(ResetOperator):
    """E: restore the environment to the untreated pre-exposure snapshot."""

    carrier = Carrier.E
    implemented = True

    def reset(self, state: BranchState, untreated: BranchState) -> BranchState:
        state.env = untreated.env.model_copy(deep=True)
        return state


class MemoryReset(ResetOperator):
    """M: snapshot-restore the note store (never a scrub -- Zombie Agents)."""

    carrier = Carrier.M
    implemented = True

    def reset(self, state: BranchState, untreated: BranchState) -> BranchState:
        import copy

        note_keys = ("plan", "notes", "memory", "lessons")
        for k in note_keys:
            if k in untreated.scaffold_state:
                state.scaffold_state[k] = copy.deepcopy(untreated.scaffold_state[k])
            elif k in state.scaffold_state:
                # untreated had no such note -> restore to empty of the same type
                cur = state.scaffold_state[k]
                state.scaffold_state[k] = [] if isinstance(cur, list) else ("" if isinstance(cur, str) else None)
        return state


_METADATA: dict[Carrier, CarrierMeta] = {
    Carrier.C: CarrierMeta(Carrier.C, "message history + KV cache", "rebuild clean context", "CV-2"),
    Carrier.M: CarrierMeta(Carrier.M, "persistent notes / lesson store", "snapshot restore", "CV-0"),
    Carrier.E: CarrierMeta(Carrier.E, "files, emails, transactions, service state", "restore env snapshot", "CV-0"),
    Carrier.B: CarrierMeta(Carrier.B, "tokens/tool-calls/steps/retries/wall-clock", "set to never-treated value", "CV-2"),
    Carrier.H: CarrierMeta(Carrier.H, "plan text, task queue, retry counters", "restore scaffold objects", "CV-2"),
    Carrier.U: CarrierMeta(Carrier.U, "caches, permissions, sessions", "restart adapters, clear caches", "CV-2"),
    Carrier.P: CarrierMeta(Carrier.P, "loaded model process, sampler state", "unload/reload same weights", "CV-2"),
    Carrier.L: CarrierMeta(Carrier.L, "intervention re-derived in agent text", "span-level redaction", "CV-2"),
}

_OPERATORS: dict[Carrier, ResetOperator] = {
    Carrier.E: EnvironmentReset(),
    Carrier.M: MemoryReset(),
    Carrier.C: CV2Placeholder(Carrier.C),
    Carrier.B: CV2Placeholder(Carrier.B),
    Carrier.H: CV2Placeholder(Carrier.H),
    Carrier.U: CV2Placeholder(Carrier.U),
    Carrier.P: CV2Placeholder(Carrier.P),
    Carrier.L: CV2Placeholder(Carrier.L),
}


def get_reset(carrier: Carrier | str) -> ResetOperator:
    c = Carrier(carrier) if isinstance(carrier, str) else carrier
    return _OPERATORS[c]


def list_carriers() -> list[Carrier]:
    return list(Carrier)


def carrier_metadata(carrier: Carrier | str) -> CarrierMeta:
    c = Carrier(carrier) if isinstance(carrier, str) else carrier
    return _METADATA[c]


__all__ = ["get_reset", "list_carriers", "carrier_metadata", "CarrierMeta", "EnvironmentReset", "MemoryReset"]
