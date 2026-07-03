"""Dataset Manager configuration.

All values are overridable via environment variables prefixed with ``DM_``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class DatasetManagerConfig(BaseSettings):
    """Configuration for the Dataset Management Platform.

    Controls dataset paths, validation parameters, split ratios,
    supported formats, and export settings.
    """

    model_config = {"env_prefix": "DM_"}

    # Base paths — relative to project root
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    datasets_root: Path = base_dir / "datasets"
    raw_dir: Path = datasets_root / "raw"
    annotated_dir: Path = datasets_root / "annotated"
    train_dir: Path = datasets_root / "train"
    validation_dir: Path = datasets_root / "validation"
    test_dir: Path = datasets_root / "test"
    exports_dir: Path = datasets_root / "exports"
    metadata_dir: Path = datasets_root / "metadata"
    versions_dir: Path = datasets_root / "versions"
    statistics_dir: Path = datasets_root / "statistics"
    quality_dir: Path = datasets_root / "quality"
    checksums_dir: Path = datasets_root / "checksums"
    logs_dir: Path = datasets_root / "logs"

    # Validation settings
    max_image_size_mb: int = 100
    supported_image_extensions: list[str] = [
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
    ]
    supported_annotation_extensions: list[str] = [
        ".json", ".xml", ".txt", ".csv",
    ]

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Default random seed for deterministic splits
    default_seed: int = 42

    # Versioning
    initial_version: str = "1.0.0"

    # Quality thresholds
    quality_min_completeness: float = 0.8
    quality_max_duplicate_pct: float = 5.0
    quality_min_validation_score: float = 0.7

    # Checksum algorithm
    checksum_algorithm: str = "sha256"


dm_config = DatasetManagerConfig()
