from __future__ import annotations

from typer.testing import CliRunner

from ticketintel import __version__
from ticketintel.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_lists_program() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ticket" in result.stdout.lower()
