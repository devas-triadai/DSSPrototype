"""Enemy Knowledge module configuration.

All values are overridable via environment variables prefixed with ``ENEMY_``.
"""

from pydantic_settings import BaseSettings


class EnemyConfig(BaseSettings):
    """Configuration for the Enemy Knowledge pipeline.

    Controls retrieval, evidence, scoring, and future vector-database
    connection parameters.
    """

    model_config = {"env_prefix": "ENEMY_"}

    # Retrieval
    retriever_type: str = "null"
    retriever_endpoint: str = ""
    retriever_timeout_seconds: float = 30.0
    max_knowledge_items: int = 50

    # Evidence
    evidence_min_weight: float = 0.1
    max_evidence_per_object: int = 10

    # Confidence
    default_confidence: float = 0.5
    confidence_weight_detection: float = 0.4
    confidence_weight_knowledge: float = 0.3
    confidence_weight_evidence: float = 0.3

    # Future — Vector Database
    vector_db_type: str = ""
    vector_db_endpoint: str = ""
    vector_db_collection: str = "enemy_knowledge"
    vector_db_timeout_seconds: float = 30.0

    # Future — Cache
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000


enemy_config = EnemyConfig()
