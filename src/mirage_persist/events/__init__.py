"""Versioned JSONL event stream (design 3.1)."""

from __future__ import annotations

from mirage_persist.events.schema import Event
from mirage_persist.events.writer import EventWriter

__all__ = ["Event", "EventWriter"]
