from __future__ import annotations

from datetime import datetime, timezone

from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import ReportGeneratorInterface
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
    QualityIssue,
    QualityReport,
    Severity,
)


class ReportGenerator(ReportGeneratorInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

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
    ) -> QualityReport:
        all_issues: list[QualityIssue] = list(score.issues) if hasattr(score, "issues") else []

        for result in [
            image_result,
            annotation_result,
            class_result,
            geometry_result,
            duplicate_result,
            outlier_result,
            imbalance_result,
            coverage_result,
            consistency_result,
            integrity_result,
        ]:
            if result:
                all_issues.extend(list(getattr(result, "issues", ())))

        error_count = sum(1 for i in all_issues if i.severity == Severity.ERROR)
        warning_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
        info_count = sum(1 for i in all_issues if i.severity == Severity.INFO)

        summary_parts: list[str] = []
        if score.production_ready:
            summary_parts.append(
                f"Dataset '{dataset_name}' is PRODUCTION READY "
                f"(score: {score.overall}/100, grade: {score.letter_grade.value})"
            )
        else:
            summary_parts.append(
                f"Dataset '{dataset_name}' is NOT production ready "
                f"(score: {score.overall}/100, grade: {score.letter_grade.value})"
            )
        summary_parts.append(f"{error_count} errors, {warning_count} warnings, {info_count} info")

        if error_count > 0:
            top_errors = [i for i in all_issues if i.severity == Severity.ERROR][:3]
            for e in top_errors:
                summary_parts.append(f"  ERROR: {e.message}")

        summary = "\n".join(summary_parts)

        return QualityReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_version=self._config.pipeline_version,
            overall_score=score,
            image_validation=image_result,
            annotation_validation=annotation_result,
            class_validation=class_result,
            geometry_validation=geometry_result,
            duplicate_detection=duplicate_result,
            outlier_detection=outlier_result,
            imbalance_analysis=imbalance_result,
            coverage_analysis=coverage_result,
            consistency_check=consistency_result,
            integrity_check=integrity_result,
            all_issues=tuple(all_issues),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            summary=summary,
        )

    async def to_json(self, report: QualityReport) -> dict[str, object]:
        return report.model_dump()

    async def to_markdown(self, report: QualityReport) -> str:
        lines: list[str] = []
        lines.append(f"# Quality Report: {report.dataset_name}")
        lines.append("")
        lines.append(f"- **Version:** {report.dataset_version}")
        lines.append(f"- **Timestamp:** {report.timestamp}")
        lines.append(f"- **Pipeline:** {report.pipeline_version}")
        lines.append("")
        lines.append("## Overall Score")
        lines.append("")
        lines.append(f"- **Score:** {report.overall_score.overall}/100")
        lines.append(f"- **Grade:** {report.overall_score.letter_grade.value}")
        lines.append(
            f"- **Production Ready:** {'Yes' if report.overall_score.production_ready else 'No'}"
        )
        lines.append("")
        lines.append("### Breakdown")
        lines.append("")
        bd = report.overall_score.breakdown
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Image Quality | {bd.image_quality}/100 |")
        lines.append(f"| Annotation Quality | {bd.annotation_quality}/100 |")
        lines.append(f"| Geometry Quality | {bd.geometry_quality}/100 |")
        lines.append(f"| Ontology Coverage | {bd.ontology_coverage}/100 |")
        lines.append(f"| Balance | {bd.balance}/100 |")
        lines.append(f"| Integrity | {bd.integrity}/100 |")
        lines.append(f"| Consistency | {bd.consistency}/100 |")
        lines.append("")
        lines.append("## Issues Summary")
        lines.append("")
        lines.append(f"- **Errors:** {report.error_count}")
        lines.append(f"- **Warnings:** {report.warning_count}")
        lines.append(f"- **Info:** {report.info_count}")
        lines.append("")
        if report.all_issues:
            lines.append("### All Issues")
            lines.append("")
            lines.append("| Severity | Category | Message | Location |")
            lines.append("|----------|----------|---------|----------|")
            for issue in report.all_issues:
                loc = issue.location or "-"
                lines.append(
                    f"| {issue.severity.value} | {issue.category.value} | {issue.message} | {loc} |"
                )
        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated by Dataset Quality Pipeline v{report.pipeline_version}*")
        return "\n".join(lines)
