from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, f1_score

from ecbias.argmining.annotator import TerminalAnnotator, agreement
from ecbias.argmining.classifier import train_and_evaluate
from ecbias.argmining.rules import default_lexicons, label_sentence
from ecbias.argmining.sentences import sentence_table

LEXICONS = default_lexicons()


def test_rules_and_priority():
    assert label_sentence("What changed?", LEXICONS)["label_rule"] == "REQUEST"
    assert label_sentence("However, margins fell.", LEXICONS)["label_rule"] == "REBUTTAL"
    assert label_sentence("Revenue was $1.2 billion.", LEXICONS)["label_rule"] == "EVIDENCE"
    assert label_sentence("We expect growth.", LEXICONS)["label_rule"] == "CLAIM"
    assert label_sentence("Welcome to the call.", LEXICONS)["label_rule"] == "OTHER"


def test_numeric_edge_cases():
    for text in ("Q3 revenue", "2024 results", "one quarter later", "$1.2 billion", "150 bps"):
        assert label_sentence(text, LEXICONS)["has_numeric"]


def test_sentence_cache_and_sides(tmp_path: Path):
    exchanges = pd.DataFrame(
        [
            {
                "exchange_id": 1,
                "question_text": "What changed? Why?",
                "answer_text": "Costs fell. Revenue grew.",
            }
        ]
    )
    result = sentence_table(exchanges, tmp_path / "sentences.parquet")
    assert set(result["side"]) == {"question", "answer"}
    assert (tmp_path / "sentences.parquet").exists()


def test_annotator_resume_and_undo(tmp_path: Path):
    path = tmp_path / "gold.csv"
    pd.DataFrame({"text": ["one", "two"], "gold_label": ["", ""]}).to_csv(path, index=False)
    annotator = TerminalAnnotator(path, "a")
    keys = iter(["1", "u", "5", "s"])
    annotator.run(input_fn=lambda _: next(keys), output_fn=lambda _: None)
    saved = pd.read_csv(path)
    assert list(saved["gold_label_a"]) == ["OTHER", "SKIP"]


def test_kappa_known_value(tmp_path: Path):
    path = tmp_path / "gold.csv"
    pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "gold_label_a": ["CLAIM", "EVIDENCE", "CLAIM", "OTHER"],
            "gold_label_b": ["CLAIM", "EVIDENCE", "OTHER", "CLAIM"],
        }
    ).to_csv(path, index=False)
    assert agreement(path, "a", "b") == cohen_kappa_score(
        ["CLAIM", "EVIDENCE", "CLAIM", "OTHER"],
        ["CLAIM", "EVIDENCE", "OTHER", "CLAIM"],
    )


def test_classifier_determinism(tmp_path: Path):
    rows = []
    for label, text in [
        ("REQUEST", "What changed?"),
        ("EVIDENCE", "Revenue was 10 percent."),
        ("CLAIM", "We expect growth."),
        ("REBUTTAL", "However, margins fell."),
        ("OTHER", "Welcome to the call."),
    ]:
        rows.extend({"text": text, "gold_label": label, "label_rule": label} for _ in range(2))
    gold = pd.DataFrame(rows)
    first = train_and_evaluate(gold, gold, tmp_path / "one", seed=42)
    second = train_and_evaluate(gold, gold, tmp_path / "two", seed=42)
    assert first["predictions"].tolist() == second["predictions"].tolist()


def test_rule_macro_f1_regression_threshold():
    gold = [
        ("What changed?", "REQUEST"),
        ("Revenue was 10 percent.", "EVIDENCE"),
        ("We expect growth.", "CLAIM"),
        ("However, margins fell.", "REBUTTAL"),
        ("Welcome to the call.", "OTHER"),
    ]
    predicted = [label_sentence(text, LEXICONS)["label_rule"] for text, _ in gold]
    actual = [label for _, label in gold]
    threshold = yaml.safe_load(Path("configs/dataset.yaml").read_text())["argmining"][
        "macro_f1_threshold"
    ]
    labels = ["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"]
    assert f1_score(actual, predicted, labels=labels, average="macro") >= threshold
