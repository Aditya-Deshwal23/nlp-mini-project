import pandas as pd

from ecbias.io.schema import TURN_COLUMNS, validate_turns
from ecbias.parsing.segmenter import segment_qa


def test_empty_transcript_and_missing_columns_are_reported():
    problems = validate_turns(pd.DataFrame())
    assert any("missing columns" in problem for problem in problems)


def test_no_qa_marker_is_excluded():
    turns = pd.DataFrame(
        [{"call_id": "c", "turn_idx": 0, "section": "unknown", "text": "Prepared remarks"}]
    )
    segmented, status = segment_qa(turns, ["question and answer"])
    assert status == {"c": False}
    assert not segmented["qa_found"].any()


def test_duplicate_ids_and_non_english_text_validate():
    values = {column: ["x", "x"] for column in TURN_COLUMNS}
    values.update({"call_id": ["c", "c"], "turn_idx": [0, 0], "text": ["你好", "こんにちは"]})
    problems = validate_turns(pd.DataFrame(values))
    assert any("duplicate" in problem for problem in problems)


def test_long_answer_does_not_change_schema():
    values = {column: ["x"] for column in TURN_COLUMNS}
    values["text"] = ["word " * 10000]
    assert validate_turns(pd.DataFrame(values)) == []
