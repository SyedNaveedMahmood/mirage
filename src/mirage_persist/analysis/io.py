"""Tabular IO (parquet, with a CSV fallback if the parquet engine is unavailable)."""

from __future__ import annotations

from pathlib import Path


def write_parquet(rows: list[dict], path: str | Path) -> Path:
    """Write rows to parquet; fall back to CSV (same stem) if no parquet engine."""
    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(p, index=False)
        return p
    except Exception:
        csv = p.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


__all__ = ["write_parquet"]
