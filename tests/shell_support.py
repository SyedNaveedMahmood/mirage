"""Helpers for testing the cluster shell libraries off-cluster.

On Linux (including the cluster) ``bash`` on PATH is the right interpreter. On a
Windows development box ``shutil.which("bash")`` usually resolves to
``C:\\Windows\\System32\\bash.exe`` or a WindowsApps alias, which launches WSL
— so it is filtered out and the usual Git-for-Windows locations are tried
instead. ``MIRAGE_TEST_BASH`` overrides the search.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"D:\Git\bin\bash.exe",
    r"D:\Git\usr\bin\bash.exe",
)


def find_bash() -> str | None:
    """Path to a POSIX bash, or None when the shell tests must be skipped."""
    override = os.environ.get("MIRAGE_TEST_BASH")
    if override:
        return override if Path(override).exists() else None

    found = shutil.which("bash")
    # WSL may also install an App Execution Alias under WindowsApps.
    if found and not any(part in found.replace("/", "\\").lower() for part in ("system32", "windowsapps")):
        return found

    for candidate in _WINDOWS_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPO_ROOT / "cluster" / "bwunicluster3"
