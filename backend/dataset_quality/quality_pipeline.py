from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.annotation_validator import AnnotationValidator
from backend.dataset_quality.class_validator import ClassValidator
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.consistency_checker import ConsistencyChecker
from backend.dataset_quality.coverage_analyzer import CoverageAnalyzer
from backend.dataset_quality.dataset_report import ReportGenerator
from backend.dataset_quality.dataset_scorer import DatasetScorer
from backend.dataset_quality.duplicate_detector import DuplicateDetector
from backend.dataset_quality.geometry_validator import GeometryValidator
from backend.dataset_quality.image_validator import ImageValidator
from backend.dataset_quality.imbalance_analyzer import ImbalanceAnalyzer
from backend.dataset_quality.integrity_checker import IntegrityChecker
from backend.dataset_quality.interfaces import (
    AnnotationValidatorInterface,
    ClassValidatorInterface,
    ConsistencyCheckerInterface,
    CoverageAnalyzerInterface,
    DatasetScorerInterface,
    DuplicateDetectorInterface,
    GeometryValidatorInterface,
    ImageValidatorInterface,
    ImbalanceAnalyzerInterface,
    IntegrityCheckerInterface,
    OutlierDetectorInterface,
    QualityPipelineInterface,
    ReportGeneratorInterface,
)
from backend.dataset_quality.models import QualityReport
from backend.dataset_quality.outlier_detector import OutlierDetector


class QualityPipeline(QualityPipelineInterface):
    def __init__(
        self,
        config: DatasetQualityConfig | None = None,
        image_validator: ImageValidatorInterface | None = None,
        annotation_validator: AnnotationValidatorInterface | None = None,
        class_validator: ClassValidatorInterface | None = None,
        geometry_validator: GeometryValidatorInterface | None = None,
        duplicate_detector: DuplicateDetectorInterface | None = None,
        outlier_detector: OutlierDetectorInterface | None = None,
        imbalance_analyzer: ImbalanceAnalyzerInterface | None = None,
        coverage_analyzer: CoverageAnalyzerInterface | None = None,
        consistency_checker: ConsistencyCheckerInterface | None = None,
        integrity_checker: IntegrityCheckerInterface | None = None,
        scorer: DatasetScorerInterface | None = None,
        report_generator: ReportGeneratorInterface | None = None,
    ):
        cfg = config or dataset_quality_config
        self._image_validator = image_validator or ImageValidator(cfg)
        self._annotation_validator = annotation_validator or AnnotationValidator()
        self._class_validator = class_validator or ClassValidator(cfg)
        self._geometry_validator = geometry_validator or GeometryValidator()
        self._duplicate_detector = duplicate_detector or DuplicateDetector(cfg)
        self._outlier_detector = outlier_detector or OutlierDetector(cfg)
        self._imbalance_analyzer = imbalance_analyzer or ImbalanceAnalyzer(cfg)
        self._coverage_analyzer = coverage_analyzer or CoverageAnalyzer()
        self._consistency_checker = consistency_checker or ConsistencyChecker()
        self._integrity_checker = integrity_checker or IntegrityChecker()
        self._scorer = scorer or DatasetScorer(cfg)
        self._report_generator = report_generator or ReportGenerator(cfg)

    async def run(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> QualityReport:
        image_result = await self._image_validator.validate(dataset.images, image_dir)
        annotation_result = await self._annotation_validator.validate(dataset)
        geometry_result = await self._geometry_validator.validate(dataset.annotations)
        class_result = await self._class_validator.validate(dataset, ontology_classes)
        duplicate_result = await self._duplicate_detector.detect(dataset, image_dir)
        outlier_result = await self._outlier_detector.detect(dataset)
        imbalance_result = await self._imbalance_analyzer.analyze(dataset)
        coverage_result = await self._coverage_analyzer.analyze(dataset, ontology_classes)
        consistency_result = await self._consistency_checker.check(dataset)
        integrity_result = await self._integrity_checker.check(dataset, image_dir)

        score = await self._scorer.score(
            image_result=image_result,
            annotation_result=annotation_result,
            geometry_result=geometry_result,
            class_result=class_result,
            duplicate_result=duplicate_result,
            outlier_result=outlier_result,
            imbalance_result=imbalance_result,
            coverage_result=coverage_result,
            consistency_result=consistency_result,
            integrity_result=integrity_result,
        )

        report = await self._report_generator.generate(
            dataset_name=dataset.name,
            dataset_version=dataset.pipeline_version,
            score=score,
            image_result=image_result,
            annotation_result=annotation_result,
            class_result=class_result,
            geometry_result=geometry_result,
            duplicate_result=duplicate_result,
            outlier_result=outlier_result,
            imbalance_result=imbalance_result,
            coverage_result=coverage_result,
            consistency_result=consistency_result,
            integrity_result=integrity_result,
        )

        return report
