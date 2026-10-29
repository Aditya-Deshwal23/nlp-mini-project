import numpy as np
import pandas as pd
import pytest

from ecbias.features.text_features import build_text_features
from ecbias.scoring.aqs import alternative_scorers, load_aqs_config, score_aqs


def _sentences(texts, labels=None):
    labels = labels or ["CLAIM"] * len(texts)
    return pd.DataFrame(
        [
            {
                "exchange_id": 1,
                "side": "question",
                "sent_idx": index,
                "text": text,
                "n_tokens": len(text.split()),
                "label": label,
            }
            for index, (text, label) in enumerate(zip(texts, labels))
        ]
    )


def test_number_never_lowers_specificity():
    base = _sentences(["We expect growth."])
    numeric = _sentences(["We expect 12 percent growth."])
    exchanges = pd.DataFrame([{"exchange_id": 1}])
    first = build_text_features(base, exchanges).iloc[0]
    second = build_text_features(numeric, exchanges).iloc[0]
    assert second.numeric_density >= first.numeric_density


def test_second_request_lowers_focus():
    one = _sentences(["What changed?"], ["REQUEST"])
    two = _sentences(["What changed?", "Why did it change?"], ["REQUEST", "REQUEST"])
    exchanges = pd.DataFrame([{"exchange_id": 1}])
    assert (
        build_text_features(two, exchanges).iloc[0].focus
        < build_text_features(one, exchanges).iloc[0].focus
    )


def test_repeated_hedge_raises_rate():
    one = _sentences(["We may grow."])
    two = _sentences(["We may may grow."])
    exchanges = pd.DataFrame([{"exchange_id": 1}])
    assert (
        build_text_features(two, exchanges).iloc[0].hedge_rate
        > build_text_features(one, exchanges).iloc[0].hedge_rate
    )


def test_empty_text_returns_nan_components():
    frame = build_text_features(_sentences([""]), pd.DataFrame([{"exchange_id": 1}]))
    assert np.isnan(frame.iloc[0].numeric_density)
    assert np.isnan(frame.iloc[0].flesch_reading_ease)


def test_bad_weights_raise(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "question:\n  weights: {structure: 0.5}\nanswer:\n  weights: {structure: 1.0}\n"
    )
    with pytest.raises(ValueError, match="sum to one"):
        load_aqs_config(path)


def test_scorers_share_index_and_scoring_is_deterministic():
    sentences = pd.concat(
        [_sentences(["What changed?"], ["REQUEST"]), _sentences(["Revenue grew."], ["EVIDENCE"])]
    ).drop_duplicates(["exchange_id", "side", "sent_idx"])
    sentences.loc[sentences.index[-1], "exchange_id"] = 2
    exchanges = pd.DataFrame([{"exchange_id": 1}, {"exchange_id": 2}])
    features = build_text_features(sentences, exchanges)
    scores_one = score_aqs(features)
    scores_two = score_aqs(features)
    pd.testing.assert_frame_equal(scores_one, scores_two)
    alternatives = alternative_scorers(
        features[features.side == "question"].set_index("exchange_id"), "question"
    )
    assert {tuple(value.index) for value in alternatives.values()} == {
        tuple(alternatives["equal"].index)
    }
