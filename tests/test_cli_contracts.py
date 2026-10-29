from typer.testing import CliRunner

from ecbias.cli import app

runner = CliRunner()


def test_commands_reject_bad_config_paths():
    for command in ("ingest", "enrich"):
        result = runner.invoke(app, [command, "--config", "missing.yaml"])
        assert result.exit_code != 0


def test_run_all_reports_missing_stage_input():
    result = runner.invoke(app, ["run-all", "--config", "missing.yaml"])
    assert result.exit_code != 0
    assert (
        "missing.yaml" in result.stdout.lower() or "missing.yaml" in str(result.exception).lower()
    )
