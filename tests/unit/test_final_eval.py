from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("yaml")
pytest.importorskip("matplotlib")

from ticketintel.config import Settings  # noqa: E402
from ticketintel.evaluation import final, report  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        novel_queue="Spam",
        dataset={"id": "a/b", "revision": "rev1", "file": "c.csv"},
        paths={
            "data_processed": tmp_path / "processed",
            "reports": tmp_path / "reports",
            "configs": tmp_path / "configs",
            "artifacts": tmp_path / "artifacts",
        },
    )


def _write_policy(settings: Settings) -> None:
    settings.paths.configs.mkdir(parents=True, exist_ok=True)
    (settings.paths.configs / "policy.yaml").write_text(
        "queue:\n"
        "  temperature: 1.3\n"
        "  abstention_threshold: 0.9\n"
        "  conformal_alpha: 0.1\n"
        "  conformal_qhat: 0.94\n",
        encoding="utf-8",
    )


def test_frozen_policy_reads_every_threshold(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_policy(settings)
    policy = final.FrozenPolicy.load(settings)
    assert policy.temperature == pytest.approx(1.3)
    assert policy.abstention_threshold == pytest.approx(0.9)
    assert policy.conformal_alpha == pytest.approx(0.1)
    assert policy.conformal_qhat == pytest.approx(0.94)


def test_label_index_maps_to_class_positions() -> None:
    classes = np.array(["billing", "tech"])
    assert list(final.label_index(classes, ["tech", "billing"])) == [1, 0]


def test_file_sha256_is_stable_and_content_dependent(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello")
    first = final.file_sha256(a)
    assert first == final.file_sha256(a)

    b = tmp_path / "b.bin"
    b.write_bytes(b"hellp")
    assert final.file_sha256(b) != first


def test_provenance_reports_missing_data(tmp_path: Path) -> None:
    prov = final.provenance(_settings(tmp_path))
    assert prov["dataset_id"] == "a/b"
    assert prov["dataset_revision"] == "rev1"
    assert prov["processed_data_sha256"] == "missing"


def test_results_json_round_trips(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = final.write_results({"queue": {"macro_f1": 0.5}}, settings)
    assert json.loads(path.read_text(encoding="utf-8")) == {"queue": {"macro_f1": 0.5}}


def _minimal_results() -> dict[str, object]:
    return {
        "queue": {
            "macro_f1": 0.5812,
            "macro_f1_ci": [0.54, 0.62],
            "min_class_recall": {"label": "Human Resources", "recall": 0.3429},
            "ece": 0.0379,
            "auto_coverage": 0.0766,
            "auto_error_rate": 0.0333,
            "aurc": 0.2046,
            "conformal_coverage": 0.8979,
            "conformal_mean_set_size": 3.246,
            "novel_queue_auroc": 0.5745,
        },
        "priority": {
            "macro_f1": 0.6169,
            "macro_f1_ci": [0.59, 0.64],
            "ordinal_mae": 0.4359,
            "ordinal_mae_ci": [0.41, 0.46],
        },
        "ablations": {
            "split_strategy": {
                "random_split_macro_f1": 0.6871,
                "group_aware_macro_f1": 0.614,
                "inflation": 0.0731,
            },
            "features": {"word_only": 0.5833, "word_plus_char": 0.578, "char_gain": -0.0054},
            "calibration_and_policy": {
                "ece_without_scaling": 0.0643,
                "ece_with_scaling": 0.0379,
                "aurc_without_scaling": 0.21,
                "aurc_with_scaling": 0.2046,
                "threshold_only_coverage": 0.0772,
                "threshold_only_error": 0.0331,
                "threshold_plus_conformal_coverage": 0.0766,
                "threshold_plus_conformal_error": 0.0333,
            },
        },
        "provenance": {"dataset_id": "a/b", "git_revision": "abc123"},
        "what_didnt_work": ["- something measurable did not help"],
    }


def test_results_md_contains_every_section_and_the_numbers() -> None:
    text = report.render_results_md(_minimal_results())
    for heading in (
        "# Results",
        "## Provenance",
        "## Queue routing (test)",
        "## Priority (test)",
        "## Ablations",
        "## Figures",
        "## What didn't work",
    ):
        assert heading in text
    assert "0.5812" in text
    assert "Human Resources" in text
    assert "- something measurable did not help" in text


def test_results_md_omits_summarization_when_absent() -> None:
    assert "## Summarization" not in report.render_results_md(_minimal_results())


def test_results_md_includes_summarization_when_present() -> None:
    results = _minimal_results()
    results["summarization"] = {
        "schema_valid_rate": 1.0,
        "unsupported_entity_rate": 0.0,
        "nli_entailed_rate": 0.42,
        "compression_ratio": 0.49,
    }
    text = report.render_results_md(results)
    assert "## Summarization" in text
    assert "0.4200" in text


def test_figures_are_written(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    y_true = ["billing", "tech", "billing", "tech"]
    y_pred = ["billing", "tech", "tech", "tech"]
    conf = np.array([0.9, 0.8, 0.6, 0.7])
    correct = np.array([1, 1, 0, 1])

    paths = [
        report.plot_confusion_matrix(y_true, y_pred, settings),
        report.plot_reliability(conf, conf, correct, settings),
        report.plot_risk_coverage({"tfidf": (conf, correct)}, settings),
        report.plot_conformal_set_size(np.array([[0.7, 0.3], [0.5, 0.5]]), 0.6, settings),
        report.plot_novelty(conf, np.array([0.2, 0.3]), 0.75, settings),
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_smoke_kpi_fixture_is_present_and_sane() -> None:
    # The CI gate compares a fresh evaluation against this frozen snapshot, so
    # a regression in the headline numbers fails the build rather than passing
    # quietly.
    fixture = Path("tests/fixtures/kpi_smoke.json")
    assert fixture.exists()
    snapshot = json.loads(fixture.read_text(encoding="utf-8"))
    assert set(snapshot) == {"queue_macro_f1", "priority_macro_f1"}
    for value in snapshot.values():
        assert 0.0 <= value <= 1.0
