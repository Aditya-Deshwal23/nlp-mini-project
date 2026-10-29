"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = (
    "paths",
    "seed",
    "dataset",
    "parsing",
    "gender",
    "argmining",
    "scoring",
    "stats",
    "viz",
)


@dataclass(frozen=True)
class PathsConfig:
    raw: str
    interim: str
    processed: str
    models: str
    results: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    format: str


@dataclass(frozen=True)
class ParsingConfig:
    question_answer_section: str
    speaker_field: str
    text_field: str


@dataclass(frozen=True)
class GenderConfig:
    source: str
    unknown_label: str


@dataclass(frozen=True)
class ArgminingConfig:
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ScoringConfig:
    scale_min: int
    scale_max: int


@dataclass(frozen=True)
class StatsConfig:
    alpha: float
    n_bootstrap: int
    n_permutations: int
    fdr_method: str


@dataclass(frozen=True)
class VizConfig:
    theme: str


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    seed: int
    dataset: DatasetConfig
    parsing: ParsingConfig
    gender: GenderConfig
    argmining: ArgminingConfig
    scoring: ScoringConfig
    stats: StatsConfig
    viz: VizConfig


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config key: {key}")
    return value


def _required(section: dict[str, Any], key: str, prefix: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required config key: {prefix}.{key}")
    return section[key]


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Config file not found: {config_path}") from error
    if not isinstance(data, dict):
        raise ValueError("Config must contain a YAML mapping")
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Missing required config key: {key}")

    paths = _section(data, "paths")
    dataset = _section(data, "dataset")
    parsing = _section(data, "parsing")
    gender = _section(data, "gender")
    argmining = _section(data, "argmining")
    scoring = _section(data, "scoring")
    stats = _section(data, "stats")
    viz = _section(data, "viz")
    return Config(
        paths=PathsConfig(
            **{
                key: _required(paths, key, "paths")
                for key in ("raw", "interim", "processed", "models", "results")
            }
        ),
        seed=data["seed"],
        dataset=DatasetConfig(
            name=_required(dataset, "name", "dataset"),
            format=_required(dataset, "format", "dataset"),
        ),
        parsing=ParsingConfig(
            question_answer_section=_required(parsing, "question_answer_section", "parsing"),
            speaker_field=_required(parsing, "speaker_field", "parsing"),
            text_field=_required(parsing, "text_field", "parsing"),
        ),
        gender=GenderConfig(
            source=_required(gender, "source", "gender"),
            unknown_label=_required(gender, "unknown_label", "gender"),
        ),
        argmining=ArgminingConfig(labels=tuple(_required(argmining, "labels", "argmining"))),
        scoring=ScoringConfig(
            scale_min=_required(scoring, "scale_min", "scoring"),
            scale_max=_required(scoring, "scale_max", "scoring"),
        ),
        stats=StatsConfig(
            alpha=_required(stats, "alpha", "stats"),
            n_bootstrap=_required(stats, "n_bootstrap", "stats"),
            n_permutations=_required(stats, "n_permutations", "stats"),
            fdr_method=_required(stats, "fdr_method", "stats"),
        ),
        viz=VizConfig(theme=_required(viz, "theme", "viz")),
    )
