"""Exchange-side text features used by the transparent AQS rubric."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import spacy
import textstat

LABELS = ("CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST")
_HEDGES = {"may", "might", "could", "possibly", "approximately", "generally", "roughly", "expect"}
_CERTAINTY = {"definitely", "clearly", "will", "must", "committed", "confident"}


def _entity_count(text: str, nlp) -> int:
    doc = nlp(text)
    entity_labels = {"PERSON", "ORG", "GPE", "MONEY", "PERCENT", "DATE", "CARDINAL", "QUANTITY"}
    return sum(entity.label_ in entity_labels for entity in doc.ents)


def build_text_features(
    sentences: pd.DataFrame,
    exchanges: pd.DataFrame,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """Return one row per exchange and side; empty text metrics are NaN."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
    rows = []
    for (exchange_id, side), group in sentences.groupby(["exchange_id", "side"], sort=False):
        texts = group["text"].fillna("").astype(str).tolist()
        joined = " ".join(texts).strip()
        tokens = re.findall(r"\b[\w'-]+\b", joined)
        n_tokens = len(tokens)
        labels = group.get(
            "label", group.get("label_rule", pd.Series(index=group.index, dtype=str))
        ).fillna("")
        n_sentences = len(group)

        def count(label: str) -> int:
            return int(labels.eq(label).sum())

        evidence_positions = [
            idx for idx, label in enumerate(labels.tolist()) if label == "EVIDENCE"
        ]
        supported = any(
            label == "CLAIM" and any(abs(idx - evidence) <= 2 for evidence in evidence_positions)
            for idx, label in enumerate(labels.tolist())
        )
        readings = [textstat.flesch_reading_ease(text) for text in texts if text.strip()]
        sentences_overlong = sum(len(re.findall(r"\b[\w'-]+\b", text)) > 35 for text in texts)
        rows.append(
            {
                "exchange_id": exchange_id,
                "side": side,
                "n_sentences": n_sentences,
                "n_tokens": n_tokens,
                "n_claim": count("CLAIM"),
                "n_evidence": count("EVIDENCE"),
                "n_rebuttal": count("REBUTTAL"),
                "n_request": count("REQUEST"),
                "claim_supported": bool(supported),
                "numeric_density": (
                    sum(any(char.isdigit() for char in token) for token in tokens) / n_tokens * 100
                )
                if n_tokens
                else float("nan"),
                "entity_density": (_entity_count(joined, nlp) / n_tokens * 100)
                if n_tokens
                else float("nan"),
                "mean_sentence_length": (n_tokens / n_sentences) if n_sentences else float("nan"),
                "flesch_reading_ease": sum(readings) / len(readings) if readings else float("nan"),
                "hedge_rate": (sum(token.lower() in _HEDGES for token in tokens) / n_tokens)
                if n_tokens
                else float("nan"),
                "certainty_rate": (sum(token.lower() in _CERTAINTY for token in tokens) / n_tokens)
                if n_tokens
                else float("nan"),
                "n_requests": count("REQUEST"),
                "is_stacked": count("REQUEST") > 1,
                "focus": 1 / max(count("REQUEST"), 1),
                "long_sentence_count": sentences_overlong,
            }
        )
    result = pd.DataFrame(rows)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
    return result
