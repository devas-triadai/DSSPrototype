"""Orchestration configuration driven by environment variables.

All values are overridable via the ``ORCH_`` prefix.
"""

from pydantic_settings import BaseSettings


class OrchestrationConfig(BaseSettings):
    """Configuration for pipeline execution, retries, and timeouts."""

    model_config = {"env_prefix": "ORCH_"}

    pipeline_timeout_seconds: float = 300.0
    default_stage_timeout_seconds: float = 60.0
    default_retry_count: int = 2
    default_retry_delay_seconds: float = 1.0
    enable_parallel_execution: bool = False


orch_config = OrchestrationConfig()
