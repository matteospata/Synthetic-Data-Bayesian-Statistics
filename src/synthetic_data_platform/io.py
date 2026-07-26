from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular input and apply basic data contract checks."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError("The MVP supports CSV input files.")
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    validate_table(frame)
    return frame


def validate_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("The input table must contain at least one row.")
    if len(frame.columns) == 0:
        raise ValueError("The input table must contain at least one column.")
    if any(not str(column).strip() for column in frame.columns):
        raise ValueError("Column names cannot be empty.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    all_null = [column for column in frame.columns if frame[column].isna().all()]
    if all_null:
        raise ValueError(f"Columns cannot be entirely null: {all_null}")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

