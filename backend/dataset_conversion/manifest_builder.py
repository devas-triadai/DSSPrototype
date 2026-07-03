from __future__ import annotations

import hashlib
import os

from backend.dataset_conversion.models import CanonicalDataset, DatasetManifest, DatasetStatistics


class ManifestBuilder:
    async def build(
        self,
        dataset: CanonicalDataset,
        statistics: DatasetStatistics | None = None,
    ) -> DatasetManifest:
        checksums: dict[str, str] = {}
        for img in dataset.images:
            if img.file_path and os.path.isfile(img.file_path):
                checksums[img.file_path] = await self._compute_checksum(img.file_path)

        return DatasetManifest(
            dataset_version=dataset.pipeline_version,
            ontology_version=dataset.ontology_version,
            source_datasets=dataset.source_datasets,
            statistics=statistics,
            checksums=checksums,
            pipeline_version=dataset.pipeline_version,
            metadata={
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "image_count": str(dataset.image_count),
                "annotation_count": str(dataset.annotation_count),
                "class_count": str(dataset.class_count),
            },
        )

    async def _compute_checksum(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return ""
