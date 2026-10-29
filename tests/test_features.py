import pandas as pd
import pytest
import yaml

from ecbias.features.pipeline import build_feature_table
from ecbias.features.response_features import build_response_features


def _exchange(question: str, answer: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exchange_id": 0,
                "call_id": "call-1",
                "call_date": "2024-02-01",
                "order_in_call": 0,
                "order_norm": 0.0,
                "analyst_name": "A. Analyst",
                "analyst_id": "abc123",
                "analyst_gender": "Unknown",
                "analyst_gender_conf": "unknown",
                "question_text": question,
                "answer_text": answer,
                "answer_turns": [{"speaker": "Executive", "text": answer}],
                "responders": ["Executive"],
                "n_responders": 1,
                "first_responder_id": "def456",
                "first_responder_role": "CFO",
                "first_responder_gender": "Unknown",
                "responder_gender_mix": "unknown",
                "is_followup": False,
                "was_interrupted": False,
            }
        ]
    )


def test_tfidf_identical_and_disjoint_texts():
    identical = build_response_features(_exchange("revenue growth", "revenue growth"))
    disjoint = build_response_features(_exchange("revenue growth", "weather forecast"))
    assert identical.loc[0, "relevance_tfidf"] == pytest.approx(1)
    assert disjoint.loc[0, "relevance_tfidf"] < 0.01


def test_openers_and_empty_question_ratio():
    courtesy = build_response_features(_exchange("", "Thanks for joining. Revenue grew."))
    direct = build_response_features(_exchange("What changed?", "Yes, revenue grew."))
    assert courtesy.loc[0, "courtesy_opener"]
    assert direct.loc[0, "direct_opener"]
    assert pd.isna(courtesy.loc[0, "answer_to_question_ratio"])


def test_operator_voice_is_not_a_courtesy_opener():
    exchanges = _exchange("How is demand?", "Thanks for joining. The operator will continue.")
    exchanges.loc[0, "answer_turns"] = [
        {"speaker": "Operator", "speaker_type": "operator", "text": exchanges.loc[0, "answer_text"]}
    ]
    frame = build_response_features(exchanges)
    assert not frame.loc[0, "courtesy_opener"]


def test_feature_spec_matches_generated_columns():
    exchanges = _exchange("What changed?", "Revenue grew 10 percent.")
    quality = pd.DataFrame(
        [
            {
                "exchange_id": 0,
                "structure_q": 0.5,
                "specificity_q": 0.5,
                "clarity_q": 0.5,
                "calibration_q": 0.5,
                "focus": 1.0,
                "AQS_q": 50.0,
                "structure_a": 0.5,
                "specificity_a": 0.5,
                "clarity_a": 0.5,
                "calibration_a": 0.5,
                "AQS_a": 50.0,
            }
        ]
    )
    result = build_feature_table(exchanges, quality)
    spec = yaml.safe_load(open("configs/feature_spec.yaml", encoding="utf-8"))
    assert set(result.columns) == {column["name"] for column in spec["columns"]}


def test_feature_row_count_equals_exchange_count():
    exchanges = pd.concat(
        [_exchange("What changed?", "Revenue grew."), _exchange("Why?", "Because.")],
        ignore_index=True,
    )
    exchanges.loc[1, "exchange_id"] = 1
    exchanges.loc[1, "call_id"] = "call-2"
    quality = pd.DataFrame(
        [
            {
                "exchange_id": exchange_id,
                "structure_q": 0.5,
                "specificity_q": 0.5,
                "clarity_q": 0.5,
                "calibration_q": 0.5,
                "focus": 1.0,
                "AQS_q": 50.0,
                "structure_a": 0.5,
                "specificity_a": 0.5,
                "clarity_a": 0.5,
                "calibration_a": 0.5,
                "AQS_a": 50.0,
            }
            for exchange_id in exchanges["exchange_id"]
        ]
    )
    result = build_feature_table(exchanges, quality)
    assert len(result) == len(exchanges)
