"""Terminal capture: tee stdout+stderr to a per-run ``terminal.log`` (UTF-8).

Required by the design: "when a run command is entered in terminal, the subsequent
outputs in terminal should be saved in a txt file" for later debugging. This is an
in-process tee so it works regardless of shell; COMMANDS.md additionally documents
PowerShell ``Start-Transcript`` as an OS-level belt-and-suspenders.

Windows note: the console is cp1252 by default, so printing unicode (sigma, epsilon,
check marks, ...) raises UnicodeEncodeError. :func:`enable_utf8_stdio` reconfigures
the console to UTF-8 with error replacement, and the tee guards writes as well.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import TextIO


def enable_utf8_stdio() -> None:
    """Make stdout/stderr tolerate UTF-8 output on a cp1252 Windows console."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class _Tee(TextIO):
    """Writes to a console stream and a logfile; tolerates console encoding errors."""

    def __init__(self, console: TextIO, logfile: TextIO) -> None:
        self._console = console
        self._logfile = logfile

    def write(self, data: str) -> int:
        try:
            self._console.write(data)
        except UnicodeEncodeError:
            enc = getattr(self._console, "encoding", "utf-8") or "utf-8"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            self._console.write(safe)
        except (ValueError, OSError):
            pass  # console closed / detached; keep logging to file
        self._logfile.write(data)
        return len(data)

    def flush(self) -> None:
        for s in (self._console, self._logfile):
            try:
                s.flush()
            except (ValueError, OSError):
                pass

    def isatty(self) -> bool:
        try:
            return bool(self._console.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._console.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._console, "encoding", "utf-8") or "utf-8"

    def writable(self) -> bool:
        return True


class TerminalCapture:
    """Context manager that tees stdout+stderr to ``log_path`` for the run's lifetime."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self._logfile: TextIO | None = None
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None

    def __enter__(self) -> TerminalCapture:
        enable_utf8_stdio()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logfile = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._logfile)  # type: ignore[assignment]
        sys.stderr = _Tee(self._orig_stderr, self._logfile)  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None


__all__ = ["TerminalCapture", "enable_utf8_stdio"]
