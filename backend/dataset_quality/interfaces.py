from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ImageInfo,
)
from backend.dataset_quality.models import (
    AnnotationValidationResult,
    ClassValidationResult,
    ConsistencyCheckResult,
    CoverageAnalysisResult,
    DatasetScore,
    DuplicateDetectionResult,
    GeometryValidationResult,
    ImageValidationResult,
    ImbalanceAnalysisResult,
    IntegrityCheckResult,
    OutlierDetectionResult,
    QualityReport,
)


class ImageValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self, images: Sequence[ImageInfo], image_dir: str | None = None
    ) -> ImageValidationResult: ...


class AnnotationValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self,
        dataset: CanonicalDataset,
    ) -> AnnotationValidationResult: ...


class ClassValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self,
        dataset: CanonicalDataset,
        ontology_classes: Sequence[str] | None = None,
    ) -> ClassValidationResult: ...


class GeometryValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self,
        annotations: Sequence[CanonicalAnnotation],
    ) -> GeometryValidationResult: ...


class DatasetValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> dict[str, object]: ...


class DuplicateDetectorInterface(ABC):
    @abstractmethod
    async def detect(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
    ) -> DuplicateDetectionResult: ...


class OutlierDetectorInterface(ABC):
    @abstractmethod
    async def detect(
        self,
        dataset: CanonicalDataset,
    ) -> OutlierDetectionResult: ...


class ImbalanceAnalyzerInterface(ABC):
    @abstractmethod
    async def analyze(
        self,
        dataset: CanonicalDataset,
    ) -> ImbalanceAnalysisResult: ...


class CoverageAnalyzerInterface(ABC):
    @abstractmethod
    async def analyze(
        self,
        dataset: CanonicalDataset,
        ontology_classes: Sequence[str] | None = None,
    ) -> CoverageAnalysisResult: ...


class ConsistencyCheckerInterface(ABC):
    @abstractmethod
    async def check(
        self,
        dataset: CanonicalDataset,
    ) -> ConsistencyCheckResult: ...


class IntegrityCheckerInterface(ABC):
    @abstractmethod
    async def check(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
    ) -> IntegrityCheckResult: ...


class DatasetScorerInterface(ABC):
    @abstractmethod
    async def score(
        self,
        image_result: ImageValidationResult | None = None,
        annotation_result: AnnotationValidationResult | None = None,
        geometry_result: GeometryValidationResult | None = None,
        class_result: ClassValidationResult | None = None,
        duplicate_result: DuplicateDetectionResult | None = None,
        outlier_result: OutlierDetectionResult | None = None,
        imbalance_result: ImbalanceAnalysisResult | None = None,
        coverage_result: CoverageAnalysisResult | None = None,
        consistency_result: ConsistencyCheckResult | None = None,
        integrity_result: IntegrityCheckResult | None = None,
    ) -> DatasetScore: ...


class ReportGeneratorInterface(ABC):
    @abstractmethod
    async def generate(
        self,
        dataset_name: str,
        dataset_version: str,
        score: DatasetScore,
        image_result: ImageValidationResult | None = None,
        annotation_result: AnnotationValidationResult | None = None,
        class_result: ClassValidationResult | None = None,
        geometry_result: GeometryValidationResult | None = None,
        duplicate_result: DuplicateDetectionResult | None = None,
        outlier_result: OutlierDetectionResult | None = None,
        imbalance_result: ImbalanceAnalysisResult | None = None,
        coverage_result: CoverageAnalysisResult | None = None,
        consistency_result: ConsistencyCheckResult | None = None,
        integrity_result: IntegrityCheckResult | None = None,
    ) -> QualityReport: ...


class QualityPipelineInterface(ABC):
    @abstractmethod
    async def run(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> QualityReport: ...
