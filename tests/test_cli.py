from typer.testing import CliRunner

from ecbias.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    commands = (
        "ingest",
        "enrich",
        "mine",
        "score",
        "features",
        "analyze",
        "figures",
        "run-all",
        "doctor",
    )
    for command in commands:
        assert command in result.stdout


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ECBias doctor" in result.stdout
    assert "[PASS]" in result.stdout or "[FAIL]" in result.stdout
