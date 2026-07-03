from __future__ import annotations

from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.exceptions import OntologyAdapterError
from backend.dataset_conversion.interfaces import OntologyAdapterInterface
from backend.ontology_mapping import (
    DatasetNotFoundError,
    MappingError,
    OntologyMappingService,
    OntologyResolver,
)


class OntologyAdapter(OntologyAdapterInterface):
    def __init__(
        self,
        mapping_service: OntologyMappingService | None = None,
        resolver: OntologyResolver | None = None,
        config: DatasetConversionConfig | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()
        self._mapping_service = mapping_service or OntologyMappingService(
            resolver=self._resolver,
        )
        self._config = config or dataset_conversion_config

    async def translate_label(
        self,
        source_label: str,
        dataset_name: str,
    ) -> str:
        try:
            result = await self._mapping_service.map_label(
                dataset_name=dataset_name,
                label=source_label,
            )
            return result.canonical_value
        except DatasetNotFoundError:
            pass
        except MappingError as e:
            if self._config.strict_mode:
                raise OntologyAdapterError(
                    f"Failed to translate label '{source_label}' for dataset '{dataset_name}': {e}"
                ) from e
            return "unknown_object"

        return "unknown_object"

    async def _try_direct_resolve(self, label: str) -> str | None:
        try:
            resolved_label = label.lower().replace(" ", "_")
            exists = await self._resolver.contains(resolved_label)
            return resolved_label if exists else None
        except Exception:
            return None

    async def translate_batch(
        self,
        source_labels: list[str],
        dataset_name: str,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for label in source_labels:
            result[label] = await self.translate_label(label, dataset_name)
        return result

    async def is_known_label(self, canonical_label: str) -> bool:
        return await self._resolver.contains(canonical_label)

    @property
    def mapping_service(self) -> OntologyMappingService:
        return self._mapping_service

    @property
    def resolver(self) -> OntologyResolver:
        return self._resolver
