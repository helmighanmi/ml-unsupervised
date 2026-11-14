# Path: tests/test_architecture.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Lightweight architecture guardrails for layered package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path("src/ml_unsupervised")

# Allowed internal import prefixes by layer (relative module depth within package).
RULES: dict[str, set[str]] = {
    "core": {"core"},
    "data": {"core", "data"},
    "algorithms": {"core", "algorithms"},
    "evaluation": {"core", "algorithms", "evaluation"},
    "workflows": {"core", "data", "algorithms", "evaluation", "workflows"},
    "reporting": {"core", "reporting"},
    "infrastructure": {"core", "infrastructure"},
    "interfaces": {
        "core",
        "data",
        "algorithms",
        "evaluation",
        "workflows",
        "reporting",
        "infrastructure",
        "interfaces",
    },
}


def _iter_layer_modules():
    for layer in RULES:
        layer_dir = PKG / layer
        if not layer_dir.is_dir():
            continue
        for path in layer_dir.rglob("*.py"):
            yield layer, path


def _imported_top_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    mod_parts = path.relative_to(PKG).parts
    # e.g. ('algorithms','clustering.py') depth 2
    depth = len(mod_parts)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level > 0:
            # Resolve relative import to package layer.
            if node.level == 1:
                # from .x import ... inside layer package means same layer
                if depth == 2:
                    found.add(mod_parts[0])
                continue
            # from ..layer import ...
            up = node.level - 1  # levels above current package
            target = node.module.split(".")[0]
            # layer modules live at depth 2; .. means package root sibling
            if depth == 2 and up == 1:
                found.add(target)
            elif up >= 2:
                # reaching above package - ignore
                continue
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ml_unsupervised."):
            found.add(node.module.split(".")[1])
    return found


def test_layer_dependency_rules() -> None:
    violations: list[str] = []
    for layer, path in _iter_layer_modules():
        for imported in sorted(_imported_top_layers(path)):
            if imported not in RULES and imported not in {"__pycache__"}:
                # ignore non-layer top-levels if any
                continue
            if imported in RULES and imported not in RULES[layer]:
                violations.append(f"{path}: layer '{layer}' must not import '{imported}'")
    assert not violations, "\n".join(violations)
