"""Tests for the Typer-based CLI."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from backend.dataset_pipeline.cli import app
from backend.dataset_pipeline.models import PipelineResult, PipelineStageResult, PipelineStatus


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_stage(name: str, status: PipelineStatus, dur: float = 0.0) -> PipelineStageResult:
    return PipelineStageResult(stage=name, status=status, duration_seconds=dur)


def _completed_stages() -> list[PipelineStageResult]:
    return [
        _make_stage("catalog", PipelineStatus.COMPLETED, 0.5),
        _make_stage("mapping", PipelineStatus.COMPLETED, 0.3),
        _make_stage("conversion", PipelineStatus.COMPLETED, 1.2),
        _make_stage("quality", PipelineStatus.COMPLETED, 0.8),
        _make_stage("training", PipelineStatus.COMPLETED, 5.0),
    ]


@pytest.fixture
def mock_service() -> MagicMock:
    svc = MagicMock()
    all_stages = [
        PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED),
        PipelineStageResult(stage="mapping", status=PipelineStatus.COMPLETED),
        PipelineStageResult(stage="conversion", status=PipelineStatus.COMPLETED),
    ]
    result = PipelineResult(
        status=PipelineStatus.COMPLETED,
        dataset_name="test_dataset",
        stages=all_stages,
    )
    svc.ingest_dataset = AsyncMock(return_value=result)
    svc.run_catalog = AsyncMock(return_value=result)
    svc.run_mapping = AsyncMock(return_value=result)
    svc.run_conversion = AsyncMock(return_value=result)
    svc.run_quality = AsyncMock(return_value=result)
    svc.run_training = AsyncMock(return_value=result)
    svc.run_full_pipeline = AsyncMock(return_value=result)
    return svc


# ------------------------------------------------------------------
# CLI command tests
# ------------------------------------------------------------------


class TestCLIIngest:
    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_basic(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "ingest", "--dataset", "coco2017", "--source", "/data/coco2017"],
        )
        assert result.exit_code == 0
        assert "COMPLETED" in result.stdout

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_skip_training(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--skip-training",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_skip_quality(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--skip-quality",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_dry_run(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_verbose(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--verbose",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_output_dir(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--output",
                "/out",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_ingest_continue_on_error(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "ingest",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--continue-on-error",
            ],
        )
        assert result.exit_code == 0


class TestCLISingleStage:
    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_catalog(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "catalog", "--dataset", "coco2017", "--source", "/data/coco2017"],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_catalog_dry_run(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "catalog",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_map(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(app, ["dataset", "map", "--dataset", "coco2017"])
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_convert(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "convert", "--dataset", "coco2017", "--source", "/data/coco2017"],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_quality(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(app, ["dataset", "quality", "--dataset", "coco2017"])
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_train(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(app, ["dataset", "train", "--dataset", "coco2017"])
        assert result.exit_code == 0


class TestCLIPipeline:
    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_pipeline_basic(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "pipeline", "--dataset", "coco2017", "--source", "/data/coco2017"],
        )
        assert result.exit_code == 0

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_pipeline_all_flags(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            [
                "dataset",
                "pipeline",
                "--dataset",
                "coco2017",
                "--source",
                "/data/coco2017",
                "--skip-quality",
                "--skip-training",
                "--continue-on-error",
                "--dry-run",
                "--verbose",
                "--output",
                "/out",
            ],
        )
        assert result.exit_code == 0


class TestCLIOutput:
    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_shows_pipeline_result(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "ingest", "--dataset", "test", "--source", "/data/test"],
        )
        assert "Pipeline Result" in result.stdout
        assert "test" in result.stdout

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_shows_stages(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "ingest", "--dataset", "test", "--source", "/data/test"],
        )
        assert "catalog" in result.stdout
        assert "mapping" in result.stdout
        assert "conversion" in result.stdout

    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_shows_summary(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(
            app,
            ["dataset", "ingest", "--dataset", "test", "--source", "/data/test"],
        )
        assert "completed" in result.stdout.lower()


class TestCLIError:
    @patch("backend.dataset_pipeline.cli.PipelineService")
    def test_missing_required_option(
        self,
        mock_svc_cls: MagicMock,
        mock_service: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_svc_cls.return_value = mock_service
        result = runner.invoke(app, ["dataset", "ingest"])
        assert result.exit_code != 0

    def test_help_output(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "dataset" in result.stdout

    def test_dataset_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["dataset", "--help"])
        assert result.exit_code == 0
        for cmd in ("ingest", "catalog", "map", "convert", "quality", "train", "pipeline"):
            assert cmd in result.stdout
