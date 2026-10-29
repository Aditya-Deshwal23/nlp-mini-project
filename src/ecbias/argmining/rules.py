"""Interpretable sentence labeller.

Primary-label priority is REQUEST > REBUTTAL > EVIDENCE > CLAIM > OTHER.
Boolean evidence flags are retained even when another label wins.
"""

from __future__ import annotations

import re
from pathlib import Path

import spacy

LABELS = ("CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER")
_NUMERIC_RE = re.compile(
    r"(?:\bq\d+\b|\b\d{4}\b|\bone quarter\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|bps?|basis points?|million|billion|quarter|year)\b"
    r"|\$\s?\d+(?:\.\d+)?\s*(?:m|bn|million|billion)?\b)",
    re.IGNORECASE,
)


def load_lexicon(path: str | Path) -> list[str]:
    return [
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _has_phrase(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _has_request(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(phrase)}\b", lowered) is not None for phrase in phrases)


def _has_numeric(text: str, nlp) -> bool:
    if _NUMERIC_RE.search(text):
        return True
    doc = nlp(text)
    return any(
        entity.label_ in {"PERCENT", "MONEY", "QUANTITY", "CARDINAL", "DATE"} for entity in doc.ents
    )


def label_sentence(text: str, lexicons: dict[str, list[str]], nlp=None) -> dict[str, bool | str]:
    """Return one primary label plus auditable trigger flags."""
    nlp = nlp or spacy.blank("en")
    is_question = "?" in text or _has_request(text, lexicons.get("request_patterns", []))
    flags = {
        "has_claim_marker": _has_phrase(text, lexicons.get("claim_markers", [])),
        "has_evidence_marker": _has_phrase(text, lexicons.get("evidence_markers", [])),
        "has_rebuttal_marker": _has_phrase(text, lexicons.get("rebuttal_markers", [])),
        "has_numeric": _has_numeric(text, nlp),
        "is_question": is_question,
    }
    if flags["is_question"]:
        label = "REQUEST"
    elif flags["has_rebuttal_marker"]:
        label = "REBUTTAL"
    elif flags["has_evidence_marker"] or flags["has_numeric"]:
        label = "EVIDENCE"
    elif flags["has_claim_marker"]:
        label = "CLAIM"
    else:
        label = "OTHER"
    return {"label_rule": label, **flags}


def default_lexicons(root: str | Path = "configs/lexicons") -> dict[str, list[str]]:
    root = Path(root)
    return {
        "claim_markers": load_lexicon(root / "claim_markers.txt"),
        "evidence_markers": load_lexicon(root / "evidence_markers.txt"),
        "rebuttal_markers": load_lexicon(root / "rebuttal_markers.txt"),
        "request_patterns": load_lexicon(root / "request_patterns.txt"),
    }
