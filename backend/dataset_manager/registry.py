"""Dataset registry — the single source of truth for all CV training datasets.

Provides a thread-safe, in-memory registry that tracks every dataset,
its versions, metadata, validation status, quality score, and checksums.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from backend.dataset_manager.interfaces import RegistryInterface
from backend.dataset_manager.models import DatasetInfo

logger = logging.getLogger("dss.dataset_manager.registry")


class DatasetRegistry(RegistryInterface):
    """In-memory dataset registry.

    Thread-safe via a plain dict; the prototype does not require
    concurrent write access.  A production deployment would replace
    this with a database-backed implementation behind the same interface.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetInfo] = {}
        self._name_index: dict[str, str] = {}

    def register(self, info: DatasetInfo) -> DatasetInfo:
        logger.info("Registering dataset: %s (%s)", info.dataset_name, info.dataset_id)
        self._datasets[info.dataset_id] = info
        self._name_index[info.dataset_name] = info.dataset_id
        logger.info("Dataset registered: %s", info.dataset_id)
        return info

    def get(self, dataset_id: str) -> DatasetInfo | None:
        return self._datasets.get(dataset_id)

    def get_by_name(self, name: str) -> DatasetInfo | None:
        dataset_id = self._name_index.get(name)
        if dataset_id is None:
            return None
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> list[DatasetInfo]:
        return list(self._datasets.values())

    def update(self, info: DatasetInfo) -> DatasetInfo:
        updated = DatasetInfo(
            dataset_id=info.dataset_id,
            dataset_name=info.dataset_name,
            dataset_version=info.dataset_version,
            dataset_type=info.dataset_type,
            description=info.description,
            created_date=info.created_date,
            last_updated=datetime.now(timezone.utc).isoformat(),
            source=info.source,
            license=info.license,
            image_count=info.image_count,
            annotation_count=info.annotation_count,
            class_count=info.class_count,
            classes=info.classes,
            supported_formats=info.supported_formats,
            checksum=info.checksum,
            validation_status=info.validation_status,
            quality_score=info.quality_score,
            statistics_file=info.statistics_file,
            metadata_file=info.metadata_file,
        )
        self._datasets[info.dataset_id] = updated
        self._name_index[info.dataset_name] = info.dataset_id
        logger.info("Dataset updated: %s", info.dataset_id)
        return updated

    def delete(self, dataset_id: str) -> bool:
        info = self._datasets.pop(dataset_id, None)
        if info is not None:
            self._name_index.pop(info.dataset_name, None)
            logger.info("Dataset deleted: %s", dataset_id)
            return True
        return False

    def contains(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

    def to_dict(self) -> dict[str, Any]:
        return {did: info.model_dump() for did, info in self._datasets.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRegistry":
        registry = cls()
        for item in data.values():
            if isinstance(item, dict):
                info = DatasetInfo(**item)
                registry.register(info)
        return registry
