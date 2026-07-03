"""Pipeline configuration.

Timeouts, logging, output paths, and default directories for the
dataset ingestion pipeline.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    model_config = {"env_prefix": "DSS_PIPELINE_"}

    catalog_source_id: str = "pipeline_ingest"
    catalog_source_type: str = "filesystem"
    catalog_curator: str = "pipeline"

    ontology_mapping_rules: list[str] = []

    conversion_output_format: str = "yolo"
    conversion_output_dir: str = "datasets/exports"

    quality_image_dir: str | None = None

    training_model_name: str = "yolov8n"
    training_batch_size: int = 16
    training_epochs: int = 100
    training_learning_rate: float = 0.001

    default_output_dir: str = "datasets/pipeline_output"

    stage_timeout_seconds: int = 3600
    pipeline_timeout_seconds: int = 28800

    continue_on_error: bool = False
    dry_run: bool = False

    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


pipeline_config = PipelineConfig()
