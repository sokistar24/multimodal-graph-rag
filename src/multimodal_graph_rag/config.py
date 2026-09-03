"""Explicit experiment configuration with a deterministic identity hash."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .schemas import content_hash

VISION_MODES = ("pixels", "captions")
EVIDENCE_CONTROLS = ("normal", "closed-book", "shuffled", "oracle", "partial-gold")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    corpus: str
    corpus_dir: str
    question_set: str
    systems: tuple[str, ...]
    generators: tuple[str, ...]
    pricing_snapshot: str
    retrieval_method: str = "dense"
    candidate_budget: int = 5
    graph_version: str = "legacy-v1"
    visual_retriever: str = "clip-vit-b32"
    judge: str = "deepseek"
    vision_judge: str = "claude-haiku"
    prompt_version: str = "legacy-v1"
    seed: int = 0
    repetition: int = 1
    bridge_hops: int = 1
    image_dir: str | None = None
    vision_mode: str = "pixels"
    controls: tuple[str, ...] = ("normal",)
    cache_dir: str = ".cache"
    results_dir: str = "artifacts/runs"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("experiment name must be non-empty")
        if self.candidate_budget < 3:
            raise ConfigurationError("candidate_budget must be at least 3")
        if not self.systems or not self.generators:
            raise ConfigurationError("systems and generators must be non-empty")
        if self.vision_mode not in VISION_MODES:
            raise ConfigurationError(f"vision_mode must be one of {VISION_MODES}")
        if self.bridge_hops not in (1, 2):
            raise ConfigurationError("bridge_hops must be 1 or 2")
        unknown = [
            control for control in self.controls if control not in EVIDENCE_CONTROLS
        ]
        if unknown:
            raise ConfigurationError(f"unknown evidence controls: {unknown}")
        if not str(self.pricing_snapshot).strip():
            raise ConfigurationError("pricing_snapshot is required")
        if self.schema_version != 1:
            raise ConfigurationError(
                f"unsupported config schema version: {self.schema_version}"
            )

    @property
    def config_hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["config_hash"] = self.config_hash
        return value

    def validate_paths(self, root: str | Path = ".") -> list[str]:
        base = Path(root)
        missing = []
        for label, value in (
            ("corpus_dir", self.corpus_dir),
            ("question_set", self.question_set),
            ("image_dir", self.image_dir),
            ("pricing_snapshot", self.pricing_snapshot),
        ):
            if value and not (base / value).exists():
                missing.append(f"{label}: {value}")
        return missing


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"config file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            return json.load(stream)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(stream)
    raise ConfigurationError(f"unsupported config format: {path.suffix}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = dict(_load_mapping(Path(path)))
    for key in ("systems", "generators", "controls"):
        if key in raw:
            raw[key] = tuple(raw[key])
    try:
        return ExperimentConfig(**raw)
    except TypeError as exc:
        raise ConfigurationError(f"invalid config {path}: {exc}") from exc
