from __future__ import annotations

import typer

from ticketintel import __version__
from ticketintel.logging import configure_logging

app = typer.Typer(add_completion=False, help="Support ticket intelligence toolkit.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
    log_level: str = typer.Option("INFO", help="Logging level."),
) -> None:
    configure_logging(log_level)


if __name__ == "__main__":
    app()
