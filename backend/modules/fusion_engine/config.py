"""Fusion Engine module configuration.

All values are overridable via environment variables prefixed with ``FUSION_``.
"""

from pydantic_settings import BaseSettings


class FusionConfig(BaseSettings):
    """Configuration for the Fusion Engine pipeline.

    Controls fusion algorithms, correlation settings, weighting
    strategies, and confidence parameters.
    """

    model_config = {"env_prefix": "FUSION_"}

    # Pipeline
    strict_validation: bool = True

    # Correlation
    correlation_threshold: float = 0.5
    max_correlations: int = 20

    # Conflict resolution
    conflict_resolution_strategy: str = "majority"
    conflict_confidence_threshold: float = 0.3

    # Confidence
    default_confidence: float = 0.5
    confidence_weight_friendly: float = 0.25
    confidence_weight_enemy: float = 0.35
    confidence_weight_terrain: float = 0.20
    confidence_weight_correlation: float = 0.20

    # Threat assessment
    threat_enemy_high_threshold: float = 0.7
    threat_friendly_mitigation: float = 0.3
    threat_terrain_amplification: float = 0.2

    # Future — Fusion algorithms
    fusion_algorithm: str = "weighted_average"
    probabilistic_model_path: str = ""

    # Future — Bayesian
    bayesian_prior_friendly: float = 0.5
    bayesian_prior_enemy: float = 0.5
    bayesian_prior_terrain: float = 0.5

    # Future — Graph
    graph_db_endpoint: str = ""
    graph_relationship_types: list[str] = ["supports", "contradicts", "corroborates"]


fusion_config = FusionConfig()
