# Path: src/ml_unsupervised/config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""YAML configuration loading with explicit validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Thin typed wrapper around project YAML configuration."""

    data: dict[str, Any]
    source: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        source = Path(path)
        if not source.exists():
            raise ConfigurationError(f"Configuration file does not exist: {source}")
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigurationError("Top-level YAML configuration must be a mapping.")
        return cls(data=loaded, source=source)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigurationError(f"Configuration section {name!r} must be a mapping.")
        return dict(value)

    def algorithm(self, section: str, name: str) -> dict[str, Any]:
        value = self.section(section).get(name, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigurationError(f"Configuration {section}.{name} must be a mapping.")
        return dict(value)


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Return raw configuration for simple scripts and notebook compatibility."""
    return ProjectConfig.from_yaml(path).data
