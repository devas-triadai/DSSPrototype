"""Priority analysis for the Decision Engine.

Assigns recommendation priority based on threat level, confidence,
and situation severity.  No recommendations.
"""

import logging
from typing import Any

from backend.contracts.models.fusion import ThreatAssessment
from backend.modules.decision_engine.config import decision_config
from backend.modules.decision_engine.interfaces import (
    PriorityAnalyzerInterface,
    SituationContext,
)

logger = logging.getLogger("dss.decision.priority_analyzer")


class PriorityAnalyzer(PriorityAnalyzerInterface):
    """Assigns recommendation priority (1 = highest, 5 = lowest).

    Base priority is determined by threat level, then adjusted
    by confidence and situation severity.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or decision_config

    def analyze(
        self,
        situation: SituationContext,
        threat: ThreatAssessment,
    ) -> int:
        """Assign a priority score (1 = highest, 5 = lowest)."""
        base = self._base_priority(threat.threat_level)

        adjustment = self._confidence_adjustment(threat.confidence)
        severity_adj = self._severity_adjustment(situation.severity)

        final = max(1, min(5, base + adjustment + severity_adj))
        logger.debug(
            "Priority: base=%d, conf_adj=%d, sev_adj=%d -> %d",
            base,
            adjustment,
            severity_adj,
            final,
        )
        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_priority(threat_level: Any) -> int:
        """Map ThreatLevel to a base priority value."""
        mapping = {
            "critical": decision_config.priority_critical,
            "high": decision_config.priority_high,
            "medium": decision_config.priority_medium,
            "low": decision_config.priority_low,
            "unknown": decision_config.priority_unknown,
        }
        key = threat_level.value if hasattr(threat_level, "value") else str(threat_level)
        return mapping.get(key, 5)

    @staticmethod
    def _confidence_adjustment(confidence: float) -> int:
        """Adjust priority based on confidence.

        Higher confidence → slightly lower priority (more certain, less urgent).
        Lower confidence → slightly higher priority (uncertainty needs attention).
        """
        if confidence >= 0.8:
            return 0
        if confidence >= 0.5:
            return 0
        return -1

    @staticmethod
    def _severity_adjustment(severity: str) -> int:
        """Adjust priority based on situation severity."""
        mapping = {
            "critical": -1,
            "high": 0,
            "medium": 0,
            "low": 1,
            "unknown": 0,
        }
        return mapping.get(severity, 0)
