"""Course-of-Action generation for the Decision Engine.

Generates possible courses of action based on the situation and
threat assessment.  The generator is configurable via environment
variables and contains no hardcoded tactical doctrine.
"""

import logging
from typing import Any

from backend.contracts.enums.core import ThreatLevel
from backend.contracts.models.fusion import ThreatAssessment
from backend.modules.decision_engine.config import decision_config
from backend.modules.decision_engine.interfaces import (
    COAGeneratorInterface,
    SituationContext,
)

logger = logging.getLogger("dss.decision.coa_generator")


class COAGenerator(COAGeneratorInterface):
    """Generates courses of action based on threat level and situation.

    COA templates are loaded from config and are fully replaceable
    via environment variables (``DECISION_COA_TEMPLATES_*``).
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or decision_config

    def generate(
        self,
        situation: SituationContext,
        threat: ThreatAssessment,
    ) -> list[str]:
        """Generate possible courses of action for the given situation."""
        templates = self._load_templates(threat.threat_level)
        if not templates:
            logger.warning("No COA templates for threat level %s", threat.threat_level.value)
            return ["Continue Surveillance"]

        actions = self._filter_relevant(templates, situation)
        logger.debug(
            "Generated %d COA(s) for threat level %s",
            len(actions),
            threat.threat_level.value,
        )
        return actions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_templates(self, threat_level: ThreatLevel) -> list[str]:
        """Load COA templates for the given threat level from config."""
        raw = self._get_template_string(threat_level)
        return [a.strip() for a in raw.split(",") if a.strip()]

    @staticmethod
    def _get_template_string(threat_level: ThreatLevel) -> str:
        """Get the raw template string for a threat level."""
        mapping = {
            ThreatLevel.CRITICAL: decision_config.coa_templates_critical,
            ThreatLevel.HIGH: decision_config.coa_templates_high,
            ThreatLevel.MEDIUM: decision_config.coa_templates_medium,
            ThreatLevel.LOW: decision_config.coa_templates_low,
            ThreatLevel.UNKNOWN: decision_config.coa_templates_unknown,
        }
        return mapping.get(threat_level, decision_config.coa_templates_unknown)

    def _filter_relevant(
        self,
        templates: list[str],
        situation: SituationContext,
    ) -> list[str]:
        """Filter COA templates based on situation relevance."""
        relevant: list[str] = []
        for action in templates:
            if self._is_relevant(action, situation):
                relevant.append(action)
        return relevant

    @staticmethod
    def _is_relevant(action: str, situation: SituationContext) -> bool:
        """Check whether a COA is relevant to the current situation."""
        action_lower = action.lower()

        if "surveillance" in action_lower and situation.severity in ("low", "unknown"):
            return True
        if "surveillance" in action_lower and situation.threat_level == ThreatLevel.HIGH:
            return True
        if "surveillance" in action_lower and situation.threat_level == ThreatLevel.CRITICAL:
            return False

        if "reinforcement" in action_lower:
            return situation.severity in ("critical", "high")
        if "headquarters" in action_lower:
            return situation.has_enemy or situation.severity != "low"
        if "reconnaissance" in action_lower:
            return situation.has_enemy
        if "defensive" in action_lower or "contingency" in action_lower:
            return situation.severity == "critical"
        if "track" in action_lower or "monitor" in action_lower:
            return True
        if "human review" in action_lower:
            return situation.threat_level == ThreatLevel.UNKNOWN
        if "log" in action_lower:
            return not situation.has_enemy

        return True
