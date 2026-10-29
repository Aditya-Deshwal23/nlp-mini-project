from pathlib import Path

from ecbias.audit import print_audit, run_audit

ROOT = Path(__file__).parents[1]


def test_audit_reports_fixture_as_clean(capsys):
    result = run_audit(ROOT)
    assert result["failures"] == []
    assert result["todos"]

    assert print_audit(ROOT) == 0
    assert "PASS: no hard audit failures" in capsys.readouterr().out


def test_audit_detects_missing_claim_and_speaker_leak(tmp_path):
    (tmp_path / "results/tables").mkdir(parents=True)
    (tmp_path / "results/tables/data_quality.csv").write_text(
        "calls_loaded,calls_with_qa_found,exchanges\n1,1,1\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("1 calls\n1 exchanges\n", encoding="utf-8")
    (tmp_path / "data/interim").mkdir(parents=True)
    import pandas as pd

    pd.DataFrame({"speaker_name": ["Alex Smith"]}).to_parquet(
        tmp_path / "data/interim/turns.parquet"
    )
    (tmp_path / "results/leak.txt").write_text("Alex Smith", encoding="utf-8")

    result = run_audit(tmp_path)
    assert any("qa calls" in failure for failure in result["failures"])
    assert any("speaker names" in failure for failure in result["failures"])
