"""Perceived gender attribution with explicit uncertainty and overrides."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from gender_guesser.detector import Detector

_DETECTOR = Detector(case_sensitive=False)
_TITLE_RE = re.compile(r"^(?:dr|mr|mrs|ms|miss)\.?\s+", re.IGNORECASE)


def first_name(name: str) -> str | None:
    value = _TITLE_RE.sub("", str(name or "").strip())
    if not value:
        return None
    token = value.split()[0].strip(".,")
    if not token or re.fullmatch(r"(?:[A-Za-z]\.){2,}", token):
        return None
    if len(token) == 1 or "." in token:
        return None
    return token


@lru_cache(maxsize=4096)
def _library_lookup(name: str) -> tuple[str, str]:
    result = _DETECTOR.get_gender(name)
    if result in {"male", "female"}:
        return ("M" if result == "male" else "F", "certain")
    if result in {"mostly_male", "mostly_female"}:
        return ("M" if result == "mostly_male" else "F", "probable")
    return "Unknown", "unknown"


def load_overrides(path: str | Path) -> dict[str, tuple[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    frame = pd.read_csv(file_path)
    required = {"name", "gender", "source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Gender overrides must contain: {', '.join(sorted(required))}")
    return {
        str(row["name"]).strip().casefold(): (
            str(row["gender"]).strip(),
            str(row["source"]).strip(),
        )
        for _, row in frame.iterrows()
    }


def infer_gender(name: str, overrides: dict[str, tuple[str, str]] | None = None) -> tuple[str, str]:
    key = str(name or "").strip().casefold()
    if overrides and key in overrides:
        return overrides[key][0], "manual"
    token = first_name(name)
    if token is None:
        return "Unknown", "unknown"
    return _library_lookup(token.casefold())


def speaker_id(name: str, salt: str) -> str:
    return hashlib.sha256((str(name) + salt).encode("utf-8")).hexdigest()[:10]
