"""Auditable speaker-role classification."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
ROLE_TYPES = {"operator", "executive", "analyst", "unknown"}
EXECUTIVE_NORMALIZED = {
    "ceo": "CEO",
    "chief executive": "CEO",
    "cfo": "CFO",
    "chief financial": "CFO",
    "coo": "other_c_suite",
    "chief operating": "other_c_suite",
    "head of investor relations": "IR",
    "investor relations": "IR",
    "ir": "IR",
}


def load_exec_titles(path: str | Path) -> set[str]:
    return {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def normalize_executive_role(raw_role: str, speaker_name: str = "") -> str:
    value = f"{raw_role} {speaker_name}".lower()
    for keyword, normalized in EXECUTIVE_NORMALIZED.items():
        if keyword in value:
            return normalized
    if any(token in value for token in ("chief ", "president", "chair", "vp", "vice president")):
        return "other"
    return "other"


def classify_speaker(
    speaker_name: str,
    raw_role: str = "",
    affiliation: str = "",
    text: str = "",
    executive_titles: set[str] | None = None,
) -> tuple[str, str]:
    """Return coarse type and normalized executive role, retaining explainable inputs."""
    value = f"{speaker_name} {raw_role} {text}".lower()
    if any(token in value for token in ("operator", "moderator", "conference call operator")):
        return "operator", ""
    titles = executive_titles or set()
    if any(title in value for title in titles):
        return "executive", normalize_executive_role(raw_role, speaker_name)
    if any(token in value for token in ("analyst", "research", "capital", "fund", "securities")):
        return "analyst", ""
    if affiliation.strip() and ("?" in text or raw_role.lower() == "analyst"):
        return "analyst", ""
    if raw_role.lower() in {"analyst", "executive", "operator"}:
        return raw_role.lower(), normalize_executive_role(
            raw_role, speaker_name
        ) if raw_role.lower() == "executive" else ""
    if "?" in text:
        return "analyst", ""
    if raw_role or speaker_name:
        return "executive", normalize_executive_role(raw_role, speaker_name)
    return "unknown", ""


def classify_turns(turns: pd.DataFrame, executive_titles: set[str]) -> pd.DataFrame:
    result = turns.copy()
    result[["speaker_type", "executive_role"]] = result.apply(
        lambda row: pd.Series(
            classify_speaker(
                str(row["speaker_name"]),
                str(row.get("speaker_raw_role", "")),
                str(row.get("speaker_affiliation", "")),
                str(row.get("text", "")),
                executive_titles,
            )
        ),
        axis=1,
    )
    unknown_share = result["speaker_type"].eq("unknown").mean() if len(result) else 0
    LOGGER.info("Unknown speaker share: %.3f", unknown_share)
    return result
