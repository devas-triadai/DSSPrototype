"""Tests for Dataset Catalog configuration."""


from backend.dataset_catalog.config import DatasetCatalogConfig, dc_config


def test_config_defaults() -> None:
    assert dc_config.weight_quality == 0.30
    assert dc_config.weight_coverage == 0.25
    assert dc_config.weight_diversity == 0.15
    assert dc_config.weight_license == 0.15
    assert dc_config.weight_source_reliability == 0.15


def test_config_env_prefix() -> None:
    cfg = DatasetCatalogConfig()
    assert cfg.model_config.get("env_prefix") == "DC_"


def test_config_paths_are_absolute() -> None:
    assert dc_config.catalog_db_path.is_absolute()
    assert dc_config.sources_db_path.is_absolute()
    assert dc_config.base_dir.exists()


def test_config_allowed_licenses() -> None:
    assert "cc0" in dc_config.allowed_licenses
    assert "mit" in dc_config.allowed_licenses
    assert "proprietary" not in dc_config.allowed_licenses


def test_config_weights_sum_to_one() -> None:
    total = (
        dc_config.weight_quality
        + dc_config.weight_coverage
        + dc_config.weight_diversity
        + dc_config.weight_license
        + dc_config.weight_source_reliability
    )
    assert abs(total - 1.0) < 0.001


def test_config_thresholds() -> None:
    assert 0.0 <= dc_config.min_taxonomy_coverage <= 1.0
    assert 0.0 <= dc_config.max_gap_severity <= 1.0
    assert dc_config.max_active_acquisitions > 0
    assert dc_config.max_pending_review_items > 0
