"""Semantic versioning for datasets.

Supports:
  - Semantic versions (v1.0.0, v1.1.0, v2.0.0)
  - Change log tracking
  - Parent version tracking
  - Version creation with checksum, validation, and statistics links
"""

import json
import logging
import re
from pathlib import Path

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import VersioningInterface
from backend.dataset_manager.models import DatasetVersion

logger = logging.getLogger("dss.dataset_manager.versioning")

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class DatasetVersioning(VersioningInterface):
    """Manages semantic versioning for datasets.

    Each version is persisted as a JSON file in the versions directory.
    """

    def __init__(self, versions_dir: Path | None = None) -> None:
        self._config = dm_config
        self._versions_dir = versions_dir or self._config.versions_dir
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        dataset_id: str,
        version: str | None = None,
        change_log: str = "",
    ) -> DatasetVersion:
        logger.info("Version creation started: %s", dataset_id)

        if version is None:
            latest = self.get_latest_version(dataset_id)
            if latest is not None:
                version = self._bump_version(latest.version)
            else:
                version = self._config.initial_version

        parent = None
        latest_parent = self.get_latest_version(dataset_id)
        if latest_parent is not None:
            parent = latest_parent.version

        new_version = DatasetVersion(
            dataset_id=dataset_id,
            version=version,
            parent_version=parent,
            change_log=change_log,
        )

        self._persist(new_version)
        logger.info("Version created: %s v%s", dataset_id, version)
        return new_version

    def get_version(self, dataset_id: str, version: str) -> DatasetVersion | None:
        path = self._version_path(dataset_id, version)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return DatasetVersion(**data)
        except Exception:
            return None

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        versions: list[DatasetVersion] = []
        pattern = f"{dataset_id}_v*.json"
        for path in sorted(self._versions_dir.glob(pattern), reverse=True):
            try:
                data = json.loads(path.read_text())
                versions.append(DatasetVersion(**data))
            except Exception:
                pass
        return versions

    def get_latest_version(self, dataset_id: str) -> DatasetVersion | None:
        versions = self.list_versions(dataset_id)
        return versions[0] if versions else None

    def _bump_version(self, current: str) -> str:
        match = _VERSION_PATTERN.match(current)
        if not match:
            return self._config.initial_version
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        patch += 1
        if patch >= 20:
            patch = 0
            minor += 1
        if minor >= 10:
            minor = 0
            major += 1
        return f"{major}.{minor}.{patch}"

    def _version_path(self, dataset_id: str, version: str) -> Path:
        return self._versions_dir / f"{dataset_id}_v{version}.json"

    def _persist(self, version: DatasetVersion) -> None:
        path = self._version_path(version.dataset_id, version.version)
        path.write_text(json.dumps(version.model_dump(), indent=2))
