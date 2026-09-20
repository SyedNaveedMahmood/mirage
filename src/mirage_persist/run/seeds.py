"""Deterministic seed schedule: ``seed = H(global_seed, checkpoint_id, branch_id, replicate)``.

This is the published seed schedule of the design (CV-0 card, "Seeds"). Seeds are
derived by hashing so that (a) they are reproducible from the schedule alone, and
(b) the same (checkpoint, branch, replicate) triple yields the same seed across
machines and runs -- which is what makes common-random-numbers (CRN) variance
reduction possible across branches sharing a checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# 63-bit mask: keep seeds in the non-negative int64 range that every backend accepts
# (numpy, torch, and the OpenAI/vLLM `seed` field all want a non-negative int).
_MASK_63 = (1 << 63) - 1


def branch_seed(
    global_seed: int,
    checkpoint_id: str,
    branch_id: str,
    replicate: int,
    *,
    salt: str = "mirage-persist/cv0",
) -> int:
    """Return the deterministic seed for one continuation replicate."""
    key = f"{salt}|{global_seed}|{checkpoint_id}|{branch_id}|{replicate}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK_63


def step_seed(base_seed: int, step: int) -> int:
    """Derive a per-decision seed from a per-continuation base seed.

    Using a distinct (but reproducible) seed per step avoids any same-seed
    correlation across decisions within one continuation while keeping the whole
    continuation reproducible from its base seed alone.
    """
    digest = hashlib.sha256(f"{base_seed}|step|{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & _MASK_63


@dataclass(frozen=True)
class SeedSchedule:
    """Bound (global_seed, salt) pair that mints per-branch seeds on demand."""

    global_seed: int
    salt: str = "mirage-persist/cv0"

    def seed_for(self, checkpoint_id: str, branch_id: str, replicate: int) -> int:
        return branch_seed(
            self.global_seed, checkpoint_id, branch_id, replicate, salt=self.salt
        )


__all__ = ["branch_seed", "step_seed", "SeedSchedule"]
