"""Version and schema-version constants for MIRAGE-Persist.

`__version__` is the package version. `SCHEMA_VERSION` is bumped independently
whenever the on-disk formats (event JSONL, run manifest, outcome parquet) change
in a backward-incompatible way, so old runs remain interpretable.
"""

from __future__ import annotations

__version__ = "0.0.1"

# On-disk schema versions (bump on breaking changes to the respective formats).
MANIFEST_SCHEMA_VERSION = "1"
EVENT_SCHEMA_VERSION = "1"
OUTCOME_SCHEMA_VERSION = "1"

# The AgentDojo revision this code is developed and pinned against. This is the
# *expected* substrate revision; the actual installed revision is recorded in
# every run manifest and compared against this at runtime (a mismatch is logged,
# not fatal, so intentional upgrades are possible).
EXPECTED_AGENTDOJO_SHA = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
EXPECTED_AGENTDOJO_VERSION = "0.1.35"
