from __future__ import annotations

from pydantic_settings import BaseSettings


class DatasetConversionConfig(BaseSettings):
    model_config = {"env_prefix": "DATASET_CONVERSION_"}

    version: str = "1.0.0"
    strict_mode: bool = True
    default_train_ratio: float = 0.7
    default_val_ratio: float = 0.15
    default_test_ratio: float = 0.15
    default_seed: int = 42
    output_dir: str = "converted_datasets"
    temp_dir: str = "temp_conversion"
    target_image_format: str = "png"
    target_color_space: str = "rgb"
    max_image_width: int = 4096
    max_image_height: int = 4096
    validate_images: bool = True
    validate_annotations: bool = True
    validate_geometry: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    plugins_enabled: bool = False
    plugins_path: str = ""


dataset_conversion_config = DatasetConversionConfig()
