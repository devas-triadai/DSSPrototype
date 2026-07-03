"""Dataset importer orchestrating format detection and parsing.

Entry point for bringing a raw dataset on disk into the DSS canonical
``RawDataset`` representation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.dataset_intelligence.exceptions import ImportError as DIImportError
from backend.dataset_intelligence.interfaces import (
    FormatParserRegistryInterface,
)
from backend.dataset_intelligence.models import ImportResult, RawDataset
from backend.dataset_intelligence.parser import FormatParserRegistry

logger = logging.getLogger("dss.dataset_intelligence.importer")


class DatasetImporter:
    """Import a raw dataset from disk into the canonical RawDataset model.

    Parameters
    ----------
    parser_registry:
        Injectable registry of format parsers. Defaults to
        ``FormatParserRegistry`` with all built-in parsers.
    """

    def __init__(
        self,
        parser_registry: FormatParserRegistryInterface | None = None,
    ) -> None:
        self._registry = parser_registry or FormatParserRegistry()

    def import_dataset(
        self,
        source_path: Path,
        dataset_name: str,
        format_hint: str | None = None,
    ) -> ImportResult:
        """Import a dataset from *source_path*.

        Returns
        -------
        ImportResult
            Contains the parsed RawDataset and a status flag.
        """
        logger.info(
            "Import started | dataset=%s | path=%s | hint=%s",
            dataset_name,
            source_path,
            format_hint,
        )
        try:
            format_name = (
                format_hint.lower() if format_hint else self._registry.detect_format(source_path)
            )
            parser = self._registry.get_parser(format_name)
            raw = parser.parse(source_path)

            # Overwrite auto-generated IDs with a stable identifier
            raw = RawDataset(
                dataset_id=f"{dataset_name}_{format_name}",
                dataset_name=dataset_name,
                import_format=format_name,
                source_path=str(source_path),
                images=raw.images,
                classes=raw.classes,
                metadata=raw.metadata,
            )

            logger.info(
                "Import complete | dataset=%s | format=%s | images=%d | classes=%d",
                dataset_name,
                format_name,
                len(raw.images),
                len(raw.classes),
            )
            return ImportResult(
                dataset_id=raw.dataset_id,
                dataset_name=dataset_name,
                import_format=format_name,
                source_path=str(source_path),
                raw_dataset=raw,
                status="validated",
            )
        except Exception as exc:
            logger.error("Import failed for %s: %s", dataset_name, exc)
            raise DIImportError(f"Import failed for {dataset_name}: {exc}") from exc
