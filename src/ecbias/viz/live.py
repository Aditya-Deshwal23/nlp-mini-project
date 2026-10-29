"""Pure, offline live-analysis helper shared by the dashboard and tests."""

from __future__ import annotations

import time

import pandas as pd

from ..argmining.rules import default_lexicons, label_sentence
from ..features.response_features import build_response_features
from ..features.text_features import build_text_features
from ..scoring.aqs import score_aqs


def analyze_text(question: str, answer: str) -> dict:
    started = time.perf_counter()
    sentences = pd.DataFrame(
        [
            {
                "exchange_id": 0,
                "side": "question",
                "sent_idx": 0,
                "text": question,
                "n_tokens": len(question.split()),
                "label": label_sentence(question, default_lexicons())["label_rule"],
            },
            {
                "exchange_id": 0,
                "side": "answer",
                "sent_idx": 0,
                "text": answer,
                "n_tokens": len(answer.split()),
                "label": label_sentence(answer, default_lexicons())["label_rule"],
            },
        ]
    )
    exchanges = pd.DataFrame(
        [
            {
                "exchange_id": 0,
                "call_id": "live",
                "question_text": question,
                "answer_text": answer,
                "answer_turns": [{"speaker": "Synthetic executive", "text": answer}],
                "responders": ["Synthetic executive"],
                "n_responders": 1,
                "first_responder_role": "other",
                "was_interrupted": False,
                "order_in_call": 0,
                "order_norm": 0.0,
                "is_followup": False,
                "analyst_name": "Synthetic analyst",
                "analyst_gender": "Unknown",
            }
        ]
    )
    text_features = build_text_features(sentences, exchanges)
    scores = score_aqs(text_features)
    response = build_response_features(exchanges)
    return {
        "sentences": sentences,
        "scores": scores.iloc[0].to_dict(),
        "response": response.iloc[0].to_dict(),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }
