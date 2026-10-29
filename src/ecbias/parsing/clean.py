"""Transcript text cleaning."""

from __future__ import annotations

import re

import pandas as pd


def clean_text(text: str, interrupted_marker: str = "[Interrupted]") -> tuple[str, bool]:
    if not text:
        return "", False
    raw = str(text)
    interrupted = raw.rstrip().endswith("--") or interrupted_marker.lower() in raw.lower()
    cleaned = raw
    while re.search(r"\[[^\[\]]*\]", cleaned):
        cleaned = re.sub(r"\[[^\[\]]*\]", " ", cleaned)
    cleaned = cleaned.translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, interrupted


def clean_turns(df: pd.DataFrame, interrupted_marker: str = "[Interrupted]") -> pd.DataFrame:
    result = df.copy()
    result["text_raw"] = result["text"]
    cleaned = result["text"].map(lambda value: clean_text(value, interrupted_marker))
    result["text"] = cleaned.map(lambda value: value[0])
    result["was_interrupted"] = cleaned.map(lambda value: value[1])
    return result
