"""Tests for pipeline configuration."""

from __future__ import annotations

from backend.dataset_pipeline.config import PipelineConfig, pipeline_config


class TestPipelineConfig:
    def test_default_values(self) -> None:
        cfg = PipelineConfig()
        assert cfg.catalog_source_id == "pipeline_ingest"
        assert cfg.catalog_source_type == "filesystem"
        assert cfg.conversion_output_format == "yolo"
        assert cfg.training_model_name == "yolov8n"
        assert cfg.training_batch_size == 16
        assert cfg.training_epochs == 100
        assert cfg.training_learning_rate == 0.001
        assert cfg.stage_timeout_seconds == 3600
        assert cfg.pipeline_timeout_seconds == 28800
        assert cfg.continue_on_error is False
        assert cfg.dry_run is False
        assert cfg.log_level == "INFO"

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(
            catalog_source_id="custom_source",
            conversion_output_format="coco",
            training_model_name="yolov8s",
            training_batch_size=32,
            training_epochs=50,
            continue_on_error=True,
            dry_run=True,
            log_level="DEBUG",
        )
        assert cfg.catalog_source_id == "custom_source"
        assert cfg.conversion_output_format == "coco"
        assert cfg.training_model_name == "yolov8s"
        assert cfg.training_batch_size == 32
        assert cfg.training_epochs == 50
        assert cfg.continue_on_error is True
        assert cfg.dry_run is True
        assert cfg.log_level == "DEBUG"

    def test_default_output_dir(self) -> None:
        cfg = PipelineConfig()
        assert cfg.default_output_dir == "datasets/pipeline_output"

    def test_stage_timeout_seconds_minimum(self) -> None:
        cfg = PipelineConfig(stage_timeout_seconds=1)
        assert cfg.stage_timeout_seconds == 1

    def test_conversion_output_dir_default(self) -> None:
        cfg = PipelineConfig()
        assert cfg.conversion_output_dir == "datasets/exports"

    def test_quality_image_dir_none_by_default(self) -> None:
        cfg = PipelineConfig()
        assert cfg.quality_image_dir is None

    def test_log_format_default(self) -> None:
        cfg = PipelineConfig()
        assert "%(asctime)s" in cfg.log_format
        assert "%(levelname)-8s" in cfg.log_format


class TestPipelineConfigSingleton:
    def test_pipeline_config_is_instance(self) -> None:
        assert isinstance(pipeline_config, PipelineConfig)

    def test_pipeline_config_has_defaults(self) -> None:
        assert pipeline_config.training_epochs == 100
