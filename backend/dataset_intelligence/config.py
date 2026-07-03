"""Dataset Intelligence configuration.

All values are overridable via environment variables prefixed with ``DI_``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class DatasetIntelligenceConfig(BaseSettings):
    """Configuration for the Dataset Intelligence & Standardization Pipeline.

    Controls import paths, processing thresholds, quality gates,
    ontology integration, and export settings.
    """

    model_config = {"env_prefix": "DI_"}

    # Base paths — relative to project root
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    datasets_root: Path = base_dir / "datasets"
    incoming_dir: Path = datasets_root / "incoming"
    processed_dir: Path = datasets_root / "processed"
    normalized_dir: Path = datasets_root / "normalized"
    harmonized_dir: Path = datasets_root / "harmonized"
    merged_dir: Path = datasets_root / "merged"
    ready_for_training_dir: Path = datasets_root / "ready_for_training"
    rejected_dir: Path = datasets_root / "rejected"
    reports_dir: Path = datasets_root / "reports"
    duplicates_dir: Path = datasets_root / "duplicates"
    imports_dir: Path = datasets_root / "imports"
    ontology_mapping_dir: Path = datasets_root / "ontology_mapping"
    exports_dir: Path = datasets_root / "exports"

    # Quality thresholds
    min_quality_score: float = 0.6
    max_duplicate_ratio: float = 0.15
    min_class_count: int = 1
    min_image_count: int = 10
    min_annotation_count: int = 10
    max_class_imbalance_ratio: float = 10.0
    min_ontology_coverage: float = 0.5

    # Duplicate detection
    near_duplicate_threshold: float = 0.95
    hash_algorithm: str = "sha256"

    # Normalization
    default_image_extension: str = ".jpg"
    target_bbox_format: str = "xywh_normalized"  # yolo format
    normalize_class_names: bool = True

    # Export
    default_export_format: str = "yolo"
    supported_export_formats: list[str] = ["yolo", "coco", "voc"]

    # Splitting
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    default_seed: int = 42
    stratified_split: bool = True

    # Provenance
    provenance_version: str = "1.0.0"

    # Parser settings
    supported_import_formats: list[str] = [
        "yolo",
        "coco",
        "voc",
        "csv",
        "json",
        "geojson",
    ]


di_config = DatasetIntelligenceConfig()
