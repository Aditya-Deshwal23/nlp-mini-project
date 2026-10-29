"""Canonical turn-table schema and validation."""

from __future__ import annotations

import pandas as pd

TURN_COLUMNS = [
    "call_id",
    "company",
    "ticker",
    "call_date",
    "section",
    "turn_idx",
    "speaker_name",
    "speaker_raw_role",
    "speaker_affiliation",
    "text",
]


def validate_turns(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    missing = [column for column in TURN_COLUMNS if column not in df.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    if "text" in df:
        if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
            problems.append("empty text found")
    if {"call_id", "turn_idx"}.issubset(df.columns):
        duplicated = df.duplicated(["call_id", "turn_idx"]).any()
        if duplicated:
            problems.append("duplicate call_id+turn_idx found")
        for call_id, group in df.groupby("call_id", sort=False):
            values = group["turn_idx"].tolist()
            if values != sorted(values):
                problems.append(f"non-monotonic turn_idx within call: {call_id}")
                break
    return problems
