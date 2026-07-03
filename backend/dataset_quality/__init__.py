from __future__ import annotations

from backend.dataset_quality.annotation_validator import AnnotationValidator
from backend.dataset_quality.class_validator import ClassValidator
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.consistency_checker import ConsistencyChecker
from backend.dataset_quality.coverage_analyzer import CoverageAnalyzer
from backend.dataset_quality.dataset_report import ReportGenerator
from backend.dataset_quality.dataset_scorer import DatasetScorer
from backend.dataset_quality.dataset_validator import DatasetValidator
from backend.dataset_quality.duplicate_detector import DuplicateDetector
from backend.dataset_quality.exceptions import (
    AnnotationValidationError,
    ClassValidationError,
    ConsistencyCheckError,
    CoverageAnalysisError,
    DuplicateDetectionError,
    GeometryValidationError,
    ImageValidationError,
    ImbalanceAnalysisError,
    IntegrityCheckError,
    OutlierDetectionError,
    PipelineError,
    QualityError,
    ReportGenerationError,
    ScoringError,
)
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
    DatasetValidatorInterface,
    DuplicateDetectorInterface,
    GeometryValidatorInterface,
    ImageValidatorInterface,
    ImbalanceAnalyzerInterface,
    IntegrityCheckerInterface,
    OutlierDetectorInterface,
    QualityPipelineInterface,
    ReportGeneratorInterface,
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
    LetterGrade,
    OutlierDetectionResult,
    QualityCategory,
    QualityIssue,
    QualityReport,
    ScoreBreakdown,
    Severity,
)
from backend.dataset_quality.outlier_detector import OutlierDetector
from backend.dataset_quality.quality_pipeline import QualityPipeline
from backend.dataset_quality.service import DatasetQualityService

__all__ = [
    "DatasetQualityConfig",
    "dataset_quality_config",
    "QualityError",
    "ImageValidationError",
    "AnnotationValidationError",
    "ClassValidationError",
    "GeometryValidationError",
    "DuplicateDetectionError",
    "OutlierDetectionError",
    "ImbalanceAnalysisError",
    "CoverageAnalysisError",
    "ConsistencyCheckError",
    "IntegrityCheckError",
    "ScoringError",
    "ReportGenerationError",
    "PipelineError",
    "ImageValidatorInterface",
    "AnnotationValidatorInterface",
    "ClassValidatorInterface",
    "GeometryValidatorInterface",
    "DatasetValidatorInterface",
    "DuplicateDetectorInterface",
    "OutlierDetectorInterface",
    "ImbalanceAnalyzerInterface",
    "CoverageAnalyzerInterface",
    "ConsistencyCheckerInterface",
    "IntegrityCheckerInterface",
    "DatasetScorerInterface",
    "ReportGeneratorInterface",
    "QualityPipelineInterface",
    "Severity",
    "QualityCategory",
    "LetterGrade",
    "QualityIssue",
    "ImageValidationResult",
    "AnnotationValidationResult",
    "ClassValidationResult",
    "GeometryValidationResult",
    "DuplicateDetectionResult",
    "OutlierDetectionResult",
    "ImbalanceAnalysisResult",
    "CoverageAnalysisResult",
    "ConsistencyCheckResult",
    "IntegrityCheckResult",
    "ScoreBreakdown",
    "DatasetScore",
    "QualityReport",
    "ImageValidator",
    "AnnotationValidator",
    "ClassValidator",
    "GeometryValidator",
    "DatasetValidator",
    "DuplicateDetector",
    "OutlierDetector",
    "ImbalanceAnalyzer",
    "CoverageAnalyzer",
    "ConsistencyChecker",
    "IntegrityChecker",
    "DatasetScorer",
    "ReportGenerator",
    "QualityPipeline",
    "DatasetQualityService",
]
