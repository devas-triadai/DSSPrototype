"""Dataset Catalog configuration.

All values are overridable via environment variables prefixed with ``DC_``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class DatasetCatalogConfig(BaseSettings):
    """Configuration for the Dataset Acquisition & Curation System.

    Controls catalog paths, scoring weights, curation workflows,
    taxonomy integration, and license policy settings.
    """

    model_config = {"env_prefix": "DC_"}

    # Base paths — relative to project root
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    catalog_db_path: Path = base_dir / "data" / "dataset_catalog" / "catalog.json"
    sources_db_path: Path = base_dir / "data" / "dataset_catalog" / "sources.json"
    profiles_dir: Path = base_dir / "data" / "dataset_catalog" / "profiles"
    taxonomy_path: Path = base_dir / "data" / "dataset_catalog" / "taxonomy.json"
    plans_dir: Path = base_dir / "data" / "dataset_catalog" / "plans"
    reports_dir: Path = base_dir / "data" / "dataset_catalog" / "reports"
    work_dir: Path = base_dir / "data" / "dataset_catalog" / "work"

    # Scoring weights (must sum to 1.0)
    weight_quality: float = 0.30
    weight_coverage: float = 0.25
    weight_diversity: float = 0.15
    weight_license: float = 0.15
    weight_source_reliability: float = 0.15

    # Coverage thresholds
    min_taxonomy_coverage: float = 0.3
    target_taxonomy_coverage: float = 0.8
    max_gap_severity: float = 0.7

    # License policy
    allowed_licenses: list[str] = [
        "cc0",
        "cc_by_4",
        "cc_by_sa_4",
        "mit",
        "apache_2",
        "odbl",
        "pddl",
    ]
    restricted_licenses: list[str] = ["cc_nc", "cc_by_nc", "unknown", "proprietary"]
    max_license_risk_score: float = 0.6

    # Curation defaults
    auto_curation_enabled: bool = False
    curation_review_required: bool = True
    max_pending_review_items: int = 100
    curation_timeout_hours: int = 168  # 7 days

    # Acquisition planning
    max_active_acquisitions: int = 5
    acquisition_budget_limit: float = 10000.0
    min_acquisition_priority_score: float = 0.4

    # Source reliability defaults
    source_reliability_decay_days: int = 180
    min_source_reliability: float = 0.1
    max_source_reliability: float = 1.0

    # Recommendation
    recommendation_limit: int = 20
    min_recommendation_score: float = 0.3


dc_config = DatasetCatalogConfig()
