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


@app.command()
def data() -> None:
    """Build the processed dataset and data card from the pinned source."""
    # Imported lazily so the CLI and its tests stay free of the ml dependencies.
    from ticketintel.config import load_settings
    from ticketintel.data import card, pipeline
    from ticketintel.data.load import load_english_tickets

    settings = load_settings()
    raw = load_english_tickets(settings)
    frame = pipeline.build(settings, raw=raw)
    processed = pipeline.save_processed(frame, settings)
    card_path = card.write(frame, settings, raw_rows=len(raw))
    typer.echo(f"processed dataset: {processed}")
    typer.echo(f"data card: {card_path}")


@app.command()
def baseline() -> None:
    """Train the TF-IDF baseline and write validation metrics to reports/interim."""
    from ticketintel.config import load_settings
    from ticketintel.evaluation import harness

    settings = load_settings()
    results = harness.run_tfidf_baseline(settings)
    path = harness.write_interim(results, settings, "baseline_tfidf_validation.json")
    typer.echo(f"baseline results: {path}")


if __name__ == "__main__":
    app()
