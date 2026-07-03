"""Environment-driven configuration for the Ontology Mapping Layer.

Every setting is prefixed with ONTOLOGY_MAPPING_ to avoid conflicts.
"""

from pydantic_settings import BaseSettings


class OntologyMappingConfig(BaseSettings):
    model_config = {"env_prefix": "ONTOLOGY_MAPPING_"}

    version: str = "1.0.0"
    strict_mode: bool = True
    logging_enabled: bool = True
    log_level: str = "INFO"
    validation_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    plugins_enabled: bool = False
    plugins_path: str = ""


ontology_mapping_config = OntologyMappingConfig()
