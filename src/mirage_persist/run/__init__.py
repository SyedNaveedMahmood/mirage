"""Per-run recording: provenance, terminal capture, seed schedule, GPU metering.

Every experiment run is wrapped in a :class:`~mirage_persist.run.run_context.RunContext`
that (1) creates a timestamped run directory, (2) tees stdout/stderr to
``terminal.log``, (3) records a full :class:`~mirage_persist.run.provenance.RunManifest`
(hardware, code SHAs, deps, config hash, model/tokenizer, seeds, GPU-hours,
peak VRAM, ...), and (4) writes the resolved config and a pip freeze.
"""

from __future__ import annotations

from mirage_persist.run.run_context import RunContext
from mirage_persist.run.seeds import SeedSchedule, branch_seed
from mirage_persist.run.terminal_capture import TerminalCapture, enable_utf8_stdio

__all__ = ["RunContext", "SeedSchedule", "branch_seed", "TerminalCapture", "enable_utf8_stdio"]
