"""Append-only JSONL event writer (UTF-8, one Event per line)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO

from mirage_persist.events.schema import Event


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class EventWriter:
    """Buffered append-only writer for the run's ``events.jsonl``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: TextIO | None = None
        self.count = 0

    def __enter__(self) -> EventWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def write(self, event: Event) -> None:
        if self._fh is None:
            raise RuntimeError("EventWriter used outside its context manager")
        self._fh.write(event.model_dump_json() + "\n")
        self.count += 1

    def emit(self, **fields: object) -> None:
        """Convenience: build an Event (ts auto-filled) and write it."""
        fields.setdefault("ts", utc_now())
        self.write(Event(**fields))  # type: ignore[arg-type]

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


__all__ = ["EventWriter", "utc_now"]
