"""Dataset Intelligence Service — public orchestrator for the full pipeline.

This is the ONLY entry point for future datasets into DSSPrototype.

Pipeline:
  Import → Detect Format → Parse → Validate → Normalize → Ontology Map →
  Detect Duplicates → Harmonize Classes → Compute Statistics → Assess Quality →
  Split → Export → Register → Handoff to Dataset Manager

All dependencies are injected. No concrete coupling.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from backend.dataset_intelligence.class_harmonizer import ClassHarmonizer
from backend.dataset_intelligence.config import di_config
from backend.dataset_intelligence.duplicate_detector import DuplicateDetector
from backend.dataset_intelligence.exceptions import (
    DatasetNotFoundError,
    ExportError,
    QualityAssessmentError,
    ValidationError,
)
from backend.dataset_intelligence.exporter import ExporterRegistry
from backend.dataset_intelligence.importer import DatasetImporter
from backend.dataset_intelligence.interfaces import (
    ClassHarmonizerInterface,
    DatasetIntelligenceRegistryInterface,
    DatasetIntelligenceServiceInterface,
    DatasetMergerInterface,
    DatasetNormalizerInterface,
    DatasetSplitterInterface,
    DatasetValidatorInterface,
    DuplicateDetectorInterface,
    OntologyMapperInterface,
    QualityEngineInterface,
    StatisticsEngineInterface,
)
from backend.dataset_intelligence.merger import DatasetMerger
from backend.dataset_intelligence.models import (
    DatasetIntelligenceRegistryEntry,
    ExportResult,
    HarmonizedDataset,
    NormalizedDataset,
    ProcessedDataset,
    QualityReport,
    StatisticsReport,
)
from backend.dataset_intelligence.normalizer import DatasetNormalizer
from backend.dataset_intelligence.ontology_mapper import OntologyMapper
from backend.dataset_intelligence.quality import QualityEngine
from backend.dataset_intelligence.registry import DatasetIntelligenceRegistry
from backend.dataset_intelligence.splitter import DatasetSplitter
from backend.dataset_intelligence.statistics import StatisticsEngine
from backend.dataset_intelligence.validator import DatasetValidator

if TYPE_CHECKING:
    from backend.dataset_manager.service import DatasetManagementService

logger = logging.getLogger("dss.dataset_intelligence.service")


class DatasetIntelligenceService(DatasetIntelligenceServiceInterface):
    """Orchestrate the full Dataset Intelligence Pipeline.

    Every constructor parameter is optional and defaults to a production-grade
    implementation. This makes the service fully testable via dependency injection.
    """

    def __init__(
        self,
        importer: DatasetImporter | None = None,
        validator: DatasetValidatorInterface | None = None,
        normalizer: DatasetNormalizerInterface | None = None,
        ontology_mapper: OntologyMapperInterface | None = None,
        duplicate_detector: DuplicateDetectorInterface | None = None,
        class_harmonizer: ClassHarmonizerInterface | None = None,
        merger: DatasetMergerInterface | None = None,
        splitter: DatasetSplitterInterface | None = None,
        statistics_engine: StatisticsEngineInterface | None = None,
        quality_engine: QualityEngineInterface | None = None,
        exporter_registry: ExporterRegistry | None = None,
        registry: DatasetIntelligenceRegistryInterface | None = None,
        dataset_manager: "DatasetManagementService | None" = None,
    ) -> None:
        self._importer = importer or DatasetImporter()
        self._validator = validator or DatasetValidator()
        self._normalizer = normalizer or DatasetNormalizer()
        self._ontology_mapper = ontology_mapper or OntologyMapper()
        self._duplicate_detector = duplicate_detector or DuplicateDetector()
        self._class_harmonizer = class_harmonizer or ClassHarmonizer()
        self._merger = merger or DatasetMerger()
        self._splitter = splitter or DatasetSplitter()
        self._statistics = statistics_engine or StatisticsEngine()
        self._quality = quality_engine or QualityEngine()
        self._exporter_registry = exporter_registry or ExporterRegistry()
        self._registry = registry or DatasetIntelligenceRegistry()
        self._dataset_manager = dataset_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        source_path: Path,
        dataset_name: str,
        format_hint: str | None = None,
    ) -> ProcessedDataset:
        """Run the full intelligence pipeline on a raw dataset.

        Returns a ``ProcessedDataset`` ready for training or registration.
        """
        logger.info("Pipeline started | dataset=%s | path=%s", dataset_name, source_path)

        # 1. Import & Parse
        import_result = self._importer.import_dataset(source_path, dataset_name, format_hint)
        raw = import_result.raw_dataset

        # 2. Validate
        validation = self._validator.validate(raw)
        if not validation.passed:
            logger.error(
                "Validation failed | dataset=%s | errors=%s", dataset_name, validation.errors
            )
            raise ValidationError(f"Validation failed for {dataset_name}: {validation.errors}")
        logger.info("Validation complete | dataset=%s | passed", dataset_name)

        # 3. Normalize
        normalized = self._normalizer.normalize(raw)
        logger.info("Normalization complete | dataset=%s", dataset_name)

        # 4. Ontology Map
        ontology_report = self._ontology_mapper.map_classes(normalized)
        normalized = self._ontology_mapper.apply_mapping(normalized, ontology_report)
        logger.info(
            "Ontology mapping complete | dataset=%s | coverage=%.2f",
            dataset_name,
            ontology_report.ontology_coverage,
        )

        # 5. Duplicate Detection
        duplicates = self._duplicate_detector.detect(normalized)
        logger.info(
            "Duplicate detection complete | dataset=%s | dups=%d | ratio=%.3f",
            dataset_name,
            len(duplicates.duplicates),
            duplicates.duplicate_ratio,
        )

        # 6. Class Harmonization
        harmonized = self._class_harmonizer.harmonize(normalized, ontology_report)
        logger.info(
            "Class harmonization complete | dataset=%s | classes=%d",
            dataset_name,
            len(harmonized.classes),
        )

        # 7. Statistics
        stats = self._statistics.compute(harmonized)
        logger.info(
            "Statistics generated | dataset=%s | images=%d | annotations=%d",
            dataset_name,
            stats.total_images,
            stats.total_annotations,
        )

        # 8. Quality Assessment
        quality = self._quality.assess(harmonized, validation, duplicates, stats)
        logger.info(
            "Quality assessment complete | dataset=%s | score=%.3f | passed=%s",
            dataset_name,
            quality.quality_score,
            quality.passed,
        )
        if not quality.passed:
            logger.error(
                "Quality gate failed | dataset=%s | errors=%s", dataset_name, quality.errors
            )
            raise QualityAssessmentError(
                f"Quality gate failed for {dataset_name}: {quality.errors}"
            )

        # 9. Split
        splits = self._splitter.split(harmonized)
        logger.info(
            "Split complete | dataset=%s | train=%d | val=%d | test=%d",
            dataset_name,
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )

        # 10. Export to default format (YOLO) in ready_for_training
        export_dir = di_config.ready_for_training_dir / dataset_name
        exporter = self._exporter_registry.get(di_config.default_export_format)
        export_result = exporter.export(
            harmonized,
            export_dir,
            class_mapping={cls: i for i, cls in enumerate(harmonized.classes)},
            splits=splits,
        )
        logger.info(
            "Export complete | dataset=%s | format=%s | dir=%s",
            dataset_name,
            export_result.format_name,
            export_result.output_dir,
        )

        # 11. Build ProcessedDataset
        processed = ProcessedDataset(
            dataset_id=harmonized.dataset_id,
            dataset_name=harmonized.dataset_name,
            version="1.0.0",
            images=harmonized.images,
            classes=harmonized.classes,
            class_mapping=export_result.class_mapping,
            splits=splits,
            statistics=stats,
            quality=quality,
            validation=validation,
            duplicates=duplicates,
            ontology_mapping=ontology_report,
            export_result=export_result,
            metadata=harmonized.metadata,
        )

        # 12. Persist reports
        self._persist_reports(processed)

        # 13. Register in DI registry
        entry = DatasetIntelligenceRegistryEntry(
            dataset_id=processed.dataset_id,
            dataset_name=processed.dataset_name,
            version=processed.version,
            import_format=raw.import_format,
            status="ready" if quality.passed else "rejected",
            quality_score=quality.quality_score,
            processed_path=str(export_dir),
            statistics_file=str(di_config.reports_dir / f"{processed.dataset_id}_stats.json"),
            quality_file=str(di_config.reports_dir / f"{processed.dataset_id}_quality.json"),
            validation_file=str(di_config.reports_dir / f"{processed.dataset_id}_validation.json"),
            report_file=str(di_config.reports_dir / f"{processed.dataset_id}_report.json"),
        )
        self._registry.register(entry)
        logger.info("Registry entry created | dataset=%s", processed.dataset_id)

        # 14. Handoff to Dataset Manager (if available, non-blocking)
        if self._dataset_manager is not None:
            try:
                self._handoff_to_dataset_manager(processed, export_dir)
                logger.info("Dataset Manager handoff complete | dataset=%s", processed.dataset_id)
            except Exception as exc:
                logger.warning("Dataset Manager handoff failed: %s", exc)

        logger.info("Pipeline complete | dataset=%s | status=ready", dataset_name)
        return processed

    def merge_datasets(
        self,
        dataset_ids: Sequence[str],
        merged_name: str,
    ) -> ProcessedDataset:
        """Merge multiple processed datasets into one unified dataset."""
        logger.info("Merge started | datasets=%s | name=%s", dataset_ids, merged_name)
        datasets: list[NormalizedDataset | HarmonizedDataset] = []
        for ds_id in dataset_ids:
            entry = self._registry.get(ds_id)
            if entry is None:
                raise DatasetNotFoundError(f"Dataset not found in registry: {ds_id}")
            # Load from persisted report if available
            report_path = Path(entry.report_file) if entry.report_file else None
            if report_path and report_path.exists():
                with report_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Reconstruct a minimal HarmonizedDataset from the report
                # In production this would deserialize fully; here we trust the report
                datasets.append(
                    HarmonizedDataset(
                        dataset_id=entry.dataset_id,
                        dataset_name=entry.dataset_name,
                        images=[],  # Would be loaded from processed_path
                        classes=raw.get("classes", []),
                    )
                )
            else:
                raise DatasetNotFoundError(f"Report file missing for dataset: {ds_id}")

        merged = self._merger.merge(datasets)
        logger.info(
            "Merge complete | merged_id=%s | images=%d", merged.dataset_id, len(merged.images)
        )

        # Re-run quality and statistics on merged dataset
        stats = self._statistics.compute(merged)
        quality = self._quality.assess(merged, None, None, stats)
        splits = self._splitter.split(merged)
        export_dir = di_config.ready_for_training_dir / merged_name
        exporter = self._exporter_registry.get(di_config.default_export_format)
        export_result = exporter.export(
            merged,
            export_dir,
            class_mapping={cls: i for i, cls in enumerate(merged.classes)},
            splits=splits,
        )

        processed = ProcessedDataset(
            dataset_id=merged.dataset_id,
            dataset_name=merged.dataset_name,
            version="1.0.0",
            images=merged.images,
            classes=merged.classes,
            class_mapping=export_result.class_mapping,
            splits=splits,
            statistics=stats,
            quality=quality,
            metadata={},
        )
        self._persist_reports(processed)
        entry = DatasetIntelligenceRegistryEntry(
            dataset_id=processed.dataset_id,
            dataset_name=processed.dataset_name,
            version=processed.version,
            import_format="merged",
            status="ready" if quality.passed else "rejected",
            quality_score=quality.quality_score,
            processed_path=str(export_dir),
        )
        self._registry.register(entry)
        return processed

    def export_dataset(
        self,
        dataset_id: str,
        format_name: str,
        output_dir: Path | None = None,
    ) -> ExportResult:
        """Export a processed dataset to a specific format."""
        entry = self._registry.get(dataset_id)
        if entry is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

        exporter = self._exporter_registry.get(format_name)
        out_dir = output_dir or (di_config.exports_dir / dataset_id / format_name)
        # Reconstruct dataset from report (simplified)
        report_path = Path(entry.report_file) if entry.report_file else None
        if report_path and report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            harmonized = HarmonizedDataset(
                dataset_id=entry.dataset_id,
                dataset_name=entry.dataset_name,
                images=[],  # Would load from disk
                classes=raw.get("classes", []),
            )
            return exporter.export(
                harmonized,
                out_dir,
                class_mapping={cls: i for i, cls in enumerate(harmonized.classes)},
            )
        raise ExportError(f"Cannot export dataset {dataset_id}: report file missing")

    def get_quality_report(self, dataset_id: str) -> QualityReport | None:
        entry = self._registry.get(dataset_id)
        if entry is None or not entry.quality_file:
            return None
        path = Path(entry.quality_file)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return QualityReport(**json.load(f))

    def get_statistics(self, dataset_id: str) -> StatisticsReport | None:
        entry = self._registry.get(dataset_id)
        if entry is None or not entry.statistics_file:
            return None
        path = Path(entry.statistics_file)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return StatisticsReport(**json.load(f))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_reports(self, processed: ProcessedDataset) -> None:
        di_config.reports_dir.mkdir(parents=True, exist_ok=True)
        if processed.statistics:
            path = di_config.reports_dir / f"{processed.dataset_id}_stats.json"
            with path.open("w", encoding="utf-8") as f:
                f.write(processed.statistics.model_dump_json(indent=2))
        if processed.quality:
            path = di_config.reports_dir / f"{processed.dataset_id}_quality.json"
            with path.open("w", encoding="utf-8") as f:
                f.write(processed.quality.model_dump_json(indent=2))
        if processed.validation:
            path = di_config.reports_dir / f"{processed.dataset_id}_validation.json"
            with path.open("w", encoding="utf-8") as f:
                f.write(processed.validation.model_dump_json(indent=2))
        report_path = di_config.reports_dir / f"{processed.dataset_id}_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(processed.model_dump_json(indent=2))

    def _handoff_to_dataset_manager(
        self,
        processed: ProcessedDataset,
        export_dir: Path,
    ) -> None:
        """Handoff a processed dataset to the existing Dataset Manager.

        This does NOT modify the Dataset Manager; it calls its public API.
        """
        from backend.dataset_manager.models import DatasetInfo, DatasetLicense

        if self._dataset_manager is None:
            return

        fmt_name = (
            processed.export_result.format_name
            if processed.export_result
            else "unknown"
        )
        info = DatasetInfo(
            dataset_id=processed.dataset_id,
            dataset_name=processed.dataset_name,
            dataset_version=processed.version,
            dataset_type="annotated",
            description=f"Processed by Dataset Intelligence Pipeline (format={fmt_name})",
            source=processed.dataset_id,
            license=DatasetLicense(),
            image_count=len(processed.images),
            annotation_count=sum(len(img.annotations) for img in processed.images),
            class_count=len(processed.classes),
            classes=processed.classes,
            validation_status="passed"
            if (processed.validation and processed.validation.passed)
            else "failed",
            quality_score=processed.quality.quality_score if processed.quality else 0.0,
        )
        self._dataset_manager.register_dataset(info)
