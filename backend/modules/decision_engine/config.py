"""Decision Engine module configuration.

All values are overridable via environment variables prefixed with ``DECISION_``.
"""

from pydantic_settings import BaseSettings


class DecisionConfig(BaseSettings):
    """Configuration for the Decision Engine pipeline.

    Controls COA generation, priority assignment, confidence
    weighting, and future doctrine / rules-engine integration.
    """

    model_config = {"env_prefix": "DECISION_"}

    # COA templates per threat level (colon-separated list)
    coa_templates_critical: str = (
        "Alert Headquarters,Request Immediate Reinforcement,"
        "Prepare Defensive Positions,Activate Contingency Plan"
    )
    coa_templates_high: str = (
        "Alert Headquarters,Request Additional Reconnaissance,"
        "Track Target,Monitor Situation"
    )
    coa_templates_medium: str = (
        "Report to Headquarters,Monitor Situation,"
        "Request Additional Reconnaissance"
    )
    coa_templates_low: str = (
        "Continue Surveillance,Log Observation"
    )
    coa_templates_unknown: str = (
        "Continue Surveillance,Request Human Review"
    )

    # Priority mapping
    priority_critical: int = 1
    priority_high: int = 2
    priority_medium: int = 3
    priority_low: int = 4
    priority_unknown: int = 5

    # Confidence
    default_confidence: float = 0.5
    confidence_weight_fusion: float = 0.40
    confidence_weight_threat: float = 0.40
    confidence_weight_situation: float = 0.20

    # Situation evaluation
    severity_high_threshold: float = 0.7
    severity_medium_threshold: float = 0.4

    # Future — Doctrine Engine
    doctrine_endpoint: str = ""
    doctrine_version: str = ""

    # Future — Rules Engine
    rules_engine_type: str = ""
    rules_engine_endpoint: str = ""

    # Future — Human Approval
    require_human_approval: bool = False
    approval_timeout_seconds: int = 300
    escalation_contact: str = ""


decision_config = DecisionConfig()
