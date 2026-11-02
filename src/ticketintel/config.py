from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# In a source checkout the configs live two levels above this module; once the
# package is installed (a container, a wheel) that path points inside
# site-packages, so the working directory is tried first.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    cwd_config = Path.cwd() / "configs" / "default.yaml"
    if cwd_config.exists():
        return cwd_config
    return _SOURCE_ROOT / "configs" / "default.yaml"


REPO_ROOT = _SOURCE_ROOT
DEFAULT_CONFIG_PATH = _default_config_path()


class Paths(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")
    configs: Path = Path("configs")


class Dataset(BaseModel):
    id: str
    revision: str
    file: str
    language: str = "en"


class Split(BaseModel):
    train: float = 0.70
    validation: float = 0.10
    calibration: float = 0.10
    test: float = 0.10


class Dedup(BaseModel):
    near_duplicate_threshold: float = 0.90


class Settings(BaseSettings):
    # Nested env overrides use the delimiter, e.g. TICKETINTEL_PATHS__REPORTS=out
    model_config = SettingsConfigDict(
        env_prefix="TICKETINTEL_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    dataset: Dataset
    split: Split = Field(default_factory=Split)
    dedup: Dedup = Field(default_factory=Dedup)
    novel_queue: str

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment variables take precedence over the YAML file so deployments
        # can override paths and seeds without editing the committed config. The
        # YAML values (carried on the instance via init) sit below env.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(data).__name__}")
    return data


def load_settings(config_path: Path | None = None) -> Settings:
    # Resolved per call so a process that changes directory still finds it.
    path = config_path or _default_config_path()
    return Settings(**_load_yaml(path))
