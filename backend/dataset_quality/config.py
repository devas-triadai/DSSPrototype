from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetQualityConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET_QUALITY_",
        env_file=".env",
        extra="ignore",
    )

    min_image_width: int = 32
    min_image_height: int = 32
    max_image_width: int = 10000
    max_image_height: int = 10000
    min_object_area: float = 4.0
    max_aspect_ratio: float = 20.0
    rare_class_threshold: float = 0.01
    duplicate_iou_threshold: float = 0.95
    near_duplicate_iou_threshold: float = 0.85
    outlier_std_dev_threshold: float = 3.0
    min_class_samples: int = 5
    output_dir: str = "quality_reports"
    strict_mode: bool = False
    pipeline_version: str = "1.0.0"


dataset_quality_config = DatasetQualityConfig()
