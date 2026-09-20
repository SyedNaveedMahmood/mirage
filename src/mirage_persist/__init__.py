"""MIRAGE-Persist: post-removal identification protocol for tool-using agents.

This package implements CV-0 (the calibration instrument) and the shared
apparatus every later experiment (CV-1..CV-5, OR-1..OR-7) reuses:

    substrate/  -- checkpoint / fork / continue engine over AgentDojo (Substrate A)
    models/     -- backend-agnostic LLM elements (mock, seeded-openai, hf-local)
    scaffolds/  -- agent scaffolds (S1 ReAct, S2 planner-executor with notes)
    interventions/ -- versioned interventions + removal operators
    resets/     -- carrier reset operators (registered here, exercised in CV-2)
    outcomes/   -- deterministic outcome scoring + subgoal decomposition
    events/     -- JSONL event stream
    stats/      -- twin-split null, bootstrap, variance components, power solver
    run/        -- per-run provenance, terminal capture, seed schedule
    experiments/cv0/ -- the four CV-0 sub-experiments + gates + report
    analysis/   -- figures and IO helpers
    config/     -- pydantic-validated, hashable experiment configs

AgentDojo is a pinned external dependency. MIRAGE subclasses its pipeline elements
and drives its suites without maintaining a fork. See docs/DEVELOPMENT.md.
"""

from __future__ import annotations

from mirage_persist.version import __version__

__all__ = ["__version__"]
