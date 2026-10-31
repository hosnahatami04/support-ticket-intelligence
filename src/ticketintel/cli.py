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


@app.command()
def train_transformer(
    time_budget_hours: float = 3.0,
    max_length: int = 256,
    train_subsample: int = 0,
) -> None:
    """Fine-tune DistilBERT and write the three-tier comparison table."""
    from ticketintel.config import load_settings
    from ticketintel.evaluation import compare, train_transformer

    settings = load_settings()
    rows = train_transformer.run(
        settings,
        time_budget_s=time_budget_hours * 3600,
        max_length=max_length,
        train_subsample=train_subsample or None,
    )
    path = compare.write_comparison(rows, settings)
    typer.echo(f"model comparison: {path}")


@app.command()
def calibrate() -> None:
    """Fit the reliability thresholds on calibration and freeze configs/policy.yaml."""
    from ticketintel.config import load_settings
    from ticketintel.reliability import calibrate as calib

    settings = load_settings()
    results = calib.calibrate_queue(settings)
    policy_path = calib.freeze_policy(results, settings)
    report_path = calib.write_report(results, settings)
    typer.echo(f"frozen policy: {policy_path}")
    typer.echo(f"calibration report: {report_path}")


@app.command()
def summarize_eval(generate: bool = True, latency: bool = True) -> None:
    """Summarise the fixed test sample and report faithfulness KPIs."""
    from ticketintel.config import load_settings
    from ticketintel.summarize import evaluate as ev

    settings = load_settings()
    cfg = ev.SummarizerSettings.load(settings)
    if generate:
        ev.generate_summaries(settings, cfg)

    results = ev.evaluate(settings, cfg)
    if latency:
        results["latency"] = ev.measure_latency(settings, cfg)
    review_rows = results["_review_rows"]
    assert isinstance(review_rows, list)
    review_path = ev.write_review_file(review_rows, settings)
    report_path = ev.write_report(results, settings)
    typer.echo(f"summarization report: {report_path}")
    typer.echo(f"nli review sample: {review_path}")


if __name__ == "__main__":
    app()
