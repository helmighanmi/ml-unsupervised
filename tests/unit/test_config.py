# Path: tests/unit/test_config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

from pathlib import Path

import pytest

from ml_unsupervised.config import ProjectConfig, load_config
from ml_unsupervised.exceptions import ConfigurationError


def test_load_config_returns_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("clustering:\n  kmeans:\n    n_clusters: 4\n", encoding="utf-8")
    data = load_config(cfg)
    assert data["clustering"]["kmeans"]["n_clusters"] == 4


def test_project_config_section_and_algorithm(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "preprocessing:\n  scaler: robust\nclustering:\n  kmeans:\n    n_clusters: 3\n",
        encoding="utf-8",
    )
    conf = ProjectConfig.from_yaml(cfg)
    assert conf.section("preprocessing")["scaler"] == "robust"
    assert conf.algorithm("clustering", "kmeans")["n_clusters"] == 3
    assert conf.section("missing") == {}


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        ProjectConfig.from_yaml(tmp_path / "nope.yaml")


def test_invalid_top_level_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        ProjectConfig.from_yaml(cfg)


def test_invalid_section_type_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("clustering: 1\n", encoding="utf-8")
    conf = ProjectConfig.from_yaml(cfg)
    with pytest.raises(ConfigurationError):
        conf.section("clustering")
