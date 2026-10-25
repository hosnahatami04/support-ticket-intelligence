from __future__ import annotations

from pathlib import Path

import pytest

from ticketintel.config import DEFAULT_CONFIG_PATH, load_settings


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 7\n"
        "dataset:\n"
        "  id: acme/tickets\n"
        "  revision: abc123\n"
        "split:\n"
        "  train: 0.6\n"
        "  validation: 0.2\n"
        "  calibration: 0.1\n"
        "  test: 0.1\n",
        encoding="utf-8",
    )
    return path


def test_default_config_loads() -> None:
    settings = load_settings()
    assert settings.dataset.id
    assert settings.dataset.revision
    assert abs(sum(vars(settings.split).values()) - 1.0) < 1e-9


def test_load_from_explicit_path(tmp_path: Path) -> None:
    settings = load_settings(_write_config(tmp_path))
    assert settings.seed == 7
    assert settings.dataset.id == "acme/tickets"
    assert settings.dataset.revision == "abc123"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TICKETINTEL_SEED", "99")
    monkeypatch.setenv("TICKETINTEL_PATHS__REPORTS", "custom_reports")
    settings = load_settings(_write_config(tmp_path))
    assert settings.seed == 99
    assert settings.paths.reports == Path("custom_reports")


def test_non_mapping_config_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_settings(path)


def test_default_config_path_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()
