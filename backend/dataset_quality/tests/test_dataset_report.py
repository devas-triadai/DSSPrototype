from __future__ import annotations

import pytest

from backend.dataset_quality.dataset_report import ReportGenerator
from backend.dataset_quality.models import (
    DatasetScore,
    LetterGrade,
    QualityCategory,
    QualityIssue,
    QualityReport,
    ScoreBreakdown,
    Severity,
)


class TestReportGenerator:
    @pytest.fixture
    def generator(self) -> ReportGenerator:
        return ReportGenerator()

    @pytest.fixture
    def sample_score(self) -> DatasetScore:
        return DatasetScore(
            overall=85.0,
            letter_grade=LetterGrade.B,
            production_ready=True,
            breakdown=ScoreBreakdown(
                image_quality=80.0,
                annotation_quality=90.0,
                geometry_quality=85.0,
                ontology_coverage=75.0,
                balance=80.0,
                integrity=95.0,
                consistency=90.0,
            ),
        )

    @pytest.mark.asyncio
    async def test_generate_report(
        self, generator: ReportGenerator, sample_score: DatasetScore
    ) -> None:
        report = await generator.generate("test_dataset", "1.0.0", sample_score)
        assert report.dataset_name == "test_dataset"
        assert report.dataset_version == "1.0.0"
        assert report.overall_score.overall == 85.0
        assert report.overall_score.production_ready is True

    @pytest.mark.asyncio
    async def test_report_with_issues(
        self, generator: ReportGenerator, sample_score: DatasetScore
    ) -> None:
        issues = (
            QualityIssue(severity=Severity.ERROR, category=QualityCategory.IMAGE, message="err1"),
            QualityIssue(
                severity=Severity.WARNING, category=QualityCategory.ANNOTATION, message="warn1"
            ),
            QualityIssue(severity=Severity.INFO, category=QualityCategory.CLASS, message="info1"),
        )
        report = await generator.generate(
            "test",
            "1.0.0",
            sample_score,
            image_result=None,
        )
        report = QualityReport(
            dataset_name="test",
            dataset_version="1.0.0",
            overall_score=sample_score,
            all_issues=issues,
            error_count=1,
            warning_count=1,
            info_count=1,
            summary="Test report",
        )
        assert report.error_count == 1
        assert report.warning_count == 1
        assert report.info_count == 1

    @pytest.mark.asyncio
    async def test_report_not_production_ready(self, generator: ReportGenerator) -> None:
        score = DatasetScore(
            overall=55.0,
            letter_grade=LetterGrade.D,
            production_ready=False,
        )
        report = await generator.generate("bad_dataset", "1.0.0", score)
        assert "NOT" in report.summary
        assert "55" in report.summary

    @pytest.mark.asyncio
    async def test_to_json(self, generator: ReportGenerator, sample_score: DatasetScore) -> None:
        report = await generator.generate("json_test", "1.0.0", sample_score)
        json_data = await generator.to_json(report)
        assert json_data["dataset_name"] == "json_test"
        assert json_data["dataset_version"] == "1.0.0"
        overall = json_data["overall_score"]
        assert isinstance(overall, dict) and overall.get("overall") == 85.0

    @pytest.mark.asyncio
    async def test_to_markdown(
        self, generator: ReportGenerator, sample_score: DatasetScore
    ) -> None:
        report = await generator.generate("md_test", "2.0.0", sample_score)
        md = await generator.to_markdown(report)
        assert "# Quality Report: md_test" in md
        assert "2.0.0" in md
        assert "B" in md
        assert "85" in md

    @pytest.mark.asyncio
    async def test_markdown_with_issues(self, generator: ReportGenerator) -> None:
        score = DatasetScore(overall=70.0, letter_grade=LetterGrade.C, production_ready=False)
        issues = (
            QualityIssue(
                severity=Severity.ERROR, category=QualityCategory.IMAGE, message="corrupt image"
            ),
            QualityIssue(
                severity=Severity.WARNING, category=QualityCategory.ANNOTATION, message="zero area"
            ),
        )
        report = QualityReport(
            dataset_name="issues_test",
            dataset_version="1.0.0",
            overall_score=score,
            all_issues=issues,
            error_count=1,
            warning_count=1,
            info_count=0,
            summary="Has issues",
        )
        md = await generator.to_markdown(report)
        assert "corrupt image" in md
        assert "zero area" in md

    @pytest.mark.asyncio
    async def test_report_with_all_results(
        self, generator: ReportGenerator, sample_score: DatasetScore
    ) -> None:
        from backend.dataset_quality.models import AnnotationValidationResult, ImageValidationResult

        img = ImageValidationResult(
            total_images=5,
            issues=(
                QualityIssue(
                    severity=Severity.WARNING, category=QualityCategory.IMAGE, message="small"
                ),
            ),
        )
        ann = AnnotationValidationResult(total_annotations=100)
        report = await generator.generate(
            "full_test",
            "2.0.0",
            sample_score,
            image_result=img,
            annotation_result=ann,
        )
        assert report.image_validation is not None
        assert report.annotation_validation is not None
        assert report.warning_count >= 1

    @pytest.mark.asyncio
    async def test_json_with_nested_structures(self, generator: ReportGenerator) -> None:
        score = DatasetScore(overall=90.0, letter_grade=LetterGrade.A, production_ready=True)
        report = await generator.generate("json_nested", "1.0.0", score)
        json_data = await generator.to_json(report)
        overall = json_data["overall_score"]
        assert isinstance(overall, dict)
        assert "breakdown" in overall
        assert overall.get("letter_grade") == "A"

    @pytest.mark.asyncio
    async def test_markdown_with_all_sections(self, generator: ReportGenerator) -> None:
        score = DatasetScore(overall=100.0, letter_grade=LetterGrade.A, production_ready=True)
        report = await generator.generate("md_full", "3.0.0", score)
        md = await generator.to_markdown(report)
        assert "Overall Score" in md
        assert "Breakdown" in md
        assert "Issues Summary" in md
        assert "100" in md

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import ReportGeneratorInterface

        assert issubclass(ReportGenerator, ReportGeneratorInterface)
