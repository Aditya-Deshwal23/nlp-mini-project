from pathlib import Path

import pandas as pd

from ecbias.io.loaders import CsvLoader, JsonlLoader, TextFolderLoader
from ecbias.io.schema import TURN_COLUMNS, validate_turns
from ecbias.parsing.clean import clean_text
from ecbias.parsing.exchanges import build_exchanges
from ecbias.parsing.segmenter import segment_qa

FIXTURES = Path("tests/fixtures")


def _config(root: Path, format_name: str, **extra):
    return {"dataset": {"root": str(root), "format": format_name, **extra}}


def test_each_loader(tmp_path: Path):
    source = {"id": "c1", "who": "Analyst One", "body": "What now?", "role": "Analyst"}
    csv_path = tmp_path / "turns.csv"
    pd.DataFrame([source]).to_csv(csv_path, index=False)
    mapping = {"call_id": "id", "speaker_name": "who", "text": "body", "speaker_raw_role": "role"}
    csv = CsvLoader(_config(tmp_path, "csv", path=str(csv_path), columns=mapping)).load()
    assert len(csv) == 1

    jsonl_path = tmp_path / "turns.jsonl"
    jsonl_path.write_text('{"id":"c1","who":"Analyst One","body":"What now?"}\n', encoding="utf-8")
    jsonl = JsonlLoader(_config(tmp_path, "jsonl", path=str(jsonl_path), columns=mapping)).load()
    assert jsonl.iloc[0]["speaker_name"] == "Analyst One"

    txt_dir = tmp_path / "text"
    txt_dir.mkdir()
    (txt_dir / "c1.txt").write_text(
        "Analyst One -- Analyst: What now?\nCEO -- Executive: We act.\n"
    )
    text = TextFolderLoader(_config(txt_dir, "text_folder")).load()
    assert len(text) == 2
    (txt_dir / "blob.txt").write_text(
        "Analyst One -- Analyst: First question. CEO -- Executive: First answer."
    )
    blob = TextFolderLoader(_config(txt_dir, "text_folder")).load()
    assert len(blob[blob["call_id"].eq("blob")]) == 2


def test_cleaning_edge_cases():
    cleaned, interrupted = clean_text("“Hello”—  [Audio gap]  world [Interrupted]")
    assert cleaned == '"Hello"- world'
    assert interrupted
    assert clean_text("before [outer [inner]] after")[0] == "before after"
    assert clean_text("")[0] == ""


def test_segmenter_with_and_without_marker():
    turns = pd.DataFrame(
        [
            {"call_id": "yes", "turn_idx": 0, "text": "Welcome to Q&A", "section": "unknown"},
            {"call_id": "no", "turn_idx": 0, "text": "Welcome", "section": "unknown"},
        ]
    )
    segmented, statuses = segment_qa(turns, ["q&a"])
    assert statuses == {"yes": True, "no": False}
    assert segmented.loc[0, "qa_found"]
    assert not segmented.loc[1, "qa_found"]


def test_exchange_rules():
    turns = pd.DataFrame(
        [
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 0,
                "speaker_name": "A",
                "speaker_type": "analyst",
                "speaker_affiliation": "X",
                "text": "First?",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 1,
                "speaker_name": "A",
                "speaker_type": "analyst",
                "speaker_affiliation": "X",
                "text": "Follow part?",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 2,
                "speaker_name": "CEO",
                "speaker_type": "executive",
                "speaker_affiliation": "",
                "text": "Answer.",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 3,
                "speaker_name": "A",
                "speaker_type": "analyst",
                "speaker_affiliation": "X",
                "text": "Follow-up?",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 4,
                "speaker_name": "CEO",
                "speaker_type": "executive",
                "speaker_affiliation": "",
                "text": "First answer.",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 5,
                "speaker_name": "CFO",
                "speaker_type": "executive",
                "speaker_affiliation": "",
                "text": "Second answer.",
            },
            {
                "call_id": "c",
                "section": "qa",
                "turn_idx": 6,
                "speaker_name": "B",
                "speaker_type": "analyst",
                "speaker_affiliation": "",
                "text": "No answer?",
            },
        ]
    )
    exchanges = build_exchanges(turns)
    assert len(exchanges) == 3
    assert exchanges.iloc[0]["n_question_turns"] == 2
    assert exchanges.iloc[1]["is_followup"]
    assert len(exchanges.iloc[1]["responders"]) == 2
    assert exchanges.iloc[2]["answer_text"] == ""


def test_validate_turns_catches_each_problem_type():
    missing = validate_turns(pd.DataFrame({"call_id": ["c"]}))
    assert any("missing columns" in problem for problem in missing)
    base = {column: ["", ""] for column in TURN_COLUMNS}
    base.update({"call_id": ["c", "c"], "turn_idx": [2, 1], "text": ["ok", ""]})
    base_df = pd.DataFrame(base)
    problems = validate_turns(base_df)
    assert any("empty text" in problem for problem in problems)
    assert any("non-monotonic" in problem for problem in problems)
    duplicate = base_df.copy()
    duplicate["turn_idx"] = [1, 1]
    assert any("duplicate" in problem for problem in validate_turns(duplicate))
