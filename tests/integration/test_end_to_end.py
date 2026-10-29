import hashlib
import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from ecbias.cli import app

runner = CliRunner()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_run_all_writes_manifest_and_stage_outputs(tmp_path, monkeypatch):
    root = Path.cwd()
    monkeypatch.chdir(tmp_path)
    for relative in ("configs", "tests/fixtures", "src"):
        source = root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
    result = runner.invoke(app, ["run-all", "--sample", "12", "--config", "configs/dataset.yaml"])
    assert result.exit_code == 0, result.stdout
    manifest = json.loads((tmp_path / "results/run_manifest.json").read_text())
    assert manifest["row_counts"]["turns"] == 38
    assert manifest["row_counts"]["exchanges"] == 12
    assert manifest["row_counts"]["features"] == 12
    assert (tmp_path / "results/figures/captions.csv").exists()


def test_fixture_outputs_are_deterministic(tmp_path, monkeypatch):
    root = Path.cwd()
    monkeypatch.chdir(tmp_path)
    for relative in ("configs", "tests/fixtures", "src"):
        source = root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
    first = runner.invoke(app, ["run-all", "--sample", "12", "--config", "configs/dataset.yaml"])
    assert first.exit_code == 0, first.stdout
    paths = [
        tmp_path / "data/processed/features.parquet",
        tmp_path / "data/processed/quality.parquet",
    ]
    before = [_digest(path) for path in paths]
    second = runner.invoke(app, ["run-all", "--sample", "12", "--config", "configs/dataset.yaml"])
    assert second.exit_code == 0, second.stdout
    assert before == [_digest(path) for path in paths]
