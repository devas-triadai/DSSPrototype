from __future__ import annotations


class QualityError(Exception):
    pass


class ImageValidationError(QualityError):
    pass


class AnnotationValidationError(QualityError):
    pass


class ClassValidationError(QualityError):
    pass


class GeometryValidationError(QualityError):
    pass


class DuplicateDetectionError(QualityError):
    pass


class OutlierDetectionError(QualityError):
    pass


class ImbalanceAnalysisError(QualityError):
    pass


class CoverageAnalysisError(QualityError):
    pass


class ConsistencyCheckError(QualityError):
    pass


class IntegrityCheckError(QualityError):
    pass


class ScoringError(QualityError):
    pass


class ReportGenerationError(QualityError):
    pass


class PipelineError(QualityError):
    pass
