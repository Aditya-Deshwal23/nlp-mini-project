"""Sentence segmentation for exchange questions and answers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import spacy


def _nlp():
    try:
        pipeline = spacy.load("en_core_web_sm")
    except OSError:
        pipeline = spacy.blank("en")
        pipeline.add_pipe("sentencizer")
    return pipeline


def sentence_table(exchanges: pd.DataFrame, output: str | Path | None = None) -> pd.DataFrame:
    """Split both sides with batched spaCy processing and optionally cache Parquet."""
    records: list[dict] = []
    texts: list[tuple[int, str, str]] = []
    for row in exchanges.itertuples():
        texts.extend(
            [
                (row.exchange_id, "question", str(row.question_text or "")),
                (row.exchange_id, "answer", str(row.answer_text or "")),
            ]
        )
    pipeline = _nlp()
    for (exchange_id, side, _), doc in zip(
        texts, pipeline.pipe((item[2] for item in texts), batch_size=64)
    ):
        for sent_idx, sentence in enumerate(doc.sents):
            text = sentence.text.strip()
            if text:
                records.append(
                    {
                        "exchange_id": exchange_id,
                        "side": side,
                        "sent_idx": sent_idx,
                        "text": text,
                        "n_tokens": len([token for token in sentence if not token.is_space]),
                    }
                )
    result = pd.DataFrame(records, columns=["exchange_id", "side", "sent_idx", "text", "n_tokens"])
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path, index=False)
    return result
