# Path: src/ml_unsupervised/infrastructure/registry.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Lightweight filesystem-based model/artifact registry."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    name: str
    version: str
    artifact_path: str
    created_at: str
    metadata: dict[str, Any]


class ModelRegistry:
    """Simple local registry storing artifacts and a JSON index.

    Intended for reproducible local tracking, not a distributed registry service.
    """

    def __init__(self, root: str | Path = "artifacts/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save_index(self, entries: list[dict[str, Any]]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")

    def register(
        self,
        name: str,
        artifact: str | Path,
        *,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        copy_artifact: bool = True,
    ) -> RegistryEntry:
        src = Path(artifact)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {src}")
        ver = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        dest_dir = self.root / name / ver
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if copy_artifact:
            shutil.copy2(src, dest)
        else:
            dest = src.resolve()
        entry = RegistryEntry(
            name=name,
            version=ver,
            artifact_path=str(dest),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )
        entries = self._load_index()
        entries.append(asdict(entry))
        self._save_index(entries)
        return entry

    def list(self, name: str | None = None) -> list[RegistryEntry]:
        entries = [RegistryEntry(**e) for e in self._load_index()]
        if name:
            entries = [e for e in entries if e.name == name]
        return entries

    def latest(self, name: str) -> RegistryEntry | None:
        items = self.list(name)
        return items[-1] if items else None
