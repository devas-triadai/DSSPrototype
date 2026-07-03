"""Metadata generator — produces a complete metadata.json for every dataset.

Includes:
  - Dataset version and creation date
  - Statistics and quality score
  - Checksum and license
  - Source and supported formats
"""

import json
import logging

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import MetadataGeneratorInterface
from backend.dataset_manager.models import (
    DatasetChecksum,
    DatasetInfo,
    DatasetMetadata,
    DatasetQuality,
    DatasetStatistics,
    DatasetValidation,
)

logger = logging.getLogger("dss.dataset_manager.metadata")


class MetadataGenerator(MetadataGeneratorInterface):
    """Generates and persists metadata.json for datasets."""

    def __init__(self) -> None:
        self._config = dm_config

    def generate(
        self,
        dataset_info: DatasetInfo,
        statistics: DatasetStatistics | None = None,
        quality: DatasetQuality | None = None,
        validation: DatasetValidation | None = None,
        checksum: DatasetChecksum | None = None,
    ) -> DatasetMetadata:
        logger.info("Metadata generation started: %s", dataset_info.dataset_id)

        metadata = DatasetMetadata(
            dataset_id=dataset_info.dataset_id,
            dataset_name=dataset_info.dataset_name,
            dataset_version=dataset_info.dataset_version,
            source=dataset_info.source,
            license=dataset_info.license,
            description=dataset_info.description,
            image_count=dataset_info.image_count,
            annotation_count=dataset_info.annotation_count,
            class_count=dataset_info.class_count,
            classes=dataset_info.classes,
            supported_formats=dataset_info.supported_formats,
            statistics=statistics,
            quality=quality,
            validation=validation,
            checksum=checksum,
        )

        self._persist(metadata)
        logger.info("Metadata generated: %s", dataset_info.dataset_id)
        return metadata

    def _persist(self, metadata: DatasetMetadata) -> None:
        meta_dir = self._config.metadata_dir
        meta_dir.mkdir(parents=True, exist_ok=True)
        path = meta_dir / f"{metadata.dataset_id}_metadata.json"
        path.write_text(json.dumps(metadata.model_dump(), indent=2, default=str))
