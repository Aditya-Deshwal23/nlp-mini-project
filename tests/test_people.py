import hashlib
from pathlib import Path

import pandas as pd

from ecbias.people.gender import first_name, infer_gender, load_overrides, speaker_id
from ecbias.people.roles import classify_speaker


def test_name_edge_cases():
    assert first_name("Dr. Priya Sharma") == "Priya"
    assert first_name("J.P. Morgan") is None
    assert first_name("Anne-Marie Smith") == "Anne-Marie"
    assert first_name("Cher") == "Cher"
    assert first_name("Élodie Martin") == "Élodie"
    assert first_name("chris") == "chris"
    assert first_name("") is None


def test_gender_override_has_priority(tmp_path: Path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([{"name": "Chris", "gender": "F", "source": "manual"}]).to_csv(path, index=False)
    assert infer_gender("chris", load_overrides(path)) == ("F", "manual")


def test_hash_stability_and_salt_change():
    assert (
        speaker_id("Priya Sharma", "salt")
        == hashlib.sha256(b"Priya Sharma" + b"salt").hexdigest()[:10]
    )
    assert speaker_id("Priya Sharma", "salt") != speaker_id("Priya Sharma", "other")


def test_roles_are_auditable_and_contextual():
    titles = {"ceo", "cfo", "head of investor relations"}
    assert (
        classify_speaker("Operator", "", "", "The next question comes from Pat", titles)[0]
        == "operator"
    )
    assert (
        classify_speaker("Pat", "Analyst", "Northstar Capital", "What changed?", titles)[0]
        == "analyst"
    )
    assert classify_speaker("Alex", "CFO", "", "We delivered.", titles) == ("executive", "CFO")
    assert (
        classify_speaker("Alex", "Analyst", "Northstar Capital", "What changed?", titles)[0]
        == "analyst"
    )


def test_results_do_not_contain_fixture_names():
    names = {"Ada Example", "Bob Sample", "Cara Demo", "Drew Fake"}
    for path in Path("results").rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            text = path.read_text(errors="ignore")
            assert all(name not in text for name in names)
