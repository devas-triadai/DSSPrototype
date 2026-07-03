"""Application settings loaded from environment variables.

Uses pydantic-settings for validation and type coercion.
All paths are relative to the project root.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for DSSPrototype.

    Every configurable value is defined here with a sensible default.
    Override any value via environment variables or a .env file.
    """

    # Application metadata
    app_name: str = "DSSPrototype"
    app_version: str = "0.1.0"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # Directory layout
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    database_path: Path = base_dir / "backend" / "database"
    upload_folder: Path = base_dir / "backend" / "uploads"
    logs_folder: Path = base_dir / "backend" / "logs"
    knowledge_base_folder: Path = base_dir / "knowledge_base"

    # Logging
    log_level: str = "DEBUG"
    log_file: str = "dss.log"

    # Model configuration (reserved for future use)
    model_cache_dir: Path = base_dir / "models" / "cache"
    model_config_path: Path = base_dir / "configs" / "models.yaml"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
