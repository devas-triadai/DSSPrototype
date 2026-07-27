"""Knowledge reasoning engine for friendly-force assessment.

Orchestrates retrieval, filtering, evidence construction, and confidence
scoring to produce a ``FriendlyAnalysis``.  Does **not** perform retrieval
or scoring itself — delegates to injected interfaces.
"""

import logging
from typing import Any

from backend.contracts.models.analysis import FriendlyAnalysis
from backend.contracts.models.detection import DetectedObject, DetectionResult
from backend.modules.knowledge.friendly.confidence_scorer import ConfidenceScorer
from backend.modules.knowledge.friendly.config import friendly_config
from backend.modules.knowledge.friendly.evidence_builder import EvidenceBuilder
from backend.modules.knowledge.friendly.interfaces import (
    ConfidenceScorerInterface,
    Evidence,
    EvidenceBuilderInterface,
    KnowledgeEngineInterface,
    KnowledgeItem,
    RetrieverInterface,
)

logger = logging.getLogger("dss.knowledge.friendly.knowledge_engine")


class KnowledgeEngine(KnowledgeEngineInterface):
    """Coordinates the friendly-knowledge assessment pipeline.

    Pipeline:
        1. Build query context from ``DetectionResult``.
        2. Retrieve knowledge for each detected object.
        3. Filter retrieved items to relevant subset.
        4. Build evidence from matches.
        5. Score final confidence.
        6. Assemble ``FriendlyAnalysis``.
    """

    def __init__(
        self,
        evidence_builder: EvidenceBuilderInterface | None = None,
        confidence_scorer: ConfidenceScorerInterface | None = None,
        config: Any | None = None,
    ) -> None:
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._config = config or friendly_config

    async def analyze(
        self,
        detection: DetectionResult,
        retriever: RetrieverInterface,
    ) -> FriendlyAnalysis:
        """Analyse a detection result using the given retriever."""
        all_evidence: list[Evidence] = []
        total_knowledge_confidence = 0.0
        matched_object_count = 0

        for obj in detection.objects:
            query = self._build_query(obj)
            context = self._build_context(detection)

            try:
                result = await retriever.retrieve(query, context)
            except Exception as exc:
                logger.warning("Retrieval failed for object %s: %s", obj.id, exc)
                continue

            relevant = self._filter_relevant(result.items, obj)
            if not relevant:
                continue

            matched_object_count += 1
            total_knowledge_confidence += max(
                (k.confidence for k in relevant), default=0.0
            )

            evidence = self._evidence_builder.build_evidence(obj, relevant)
            all_evidence.extend(evidence)

        has_match = matched_object_count > 0

        if matched_object_count > 0:
            avg_detection_conf = self._average_detection_confidence(detection)
            avg_knowledge_conf = total_knowledge_confidence / matched_object_count
        else:
            avg_detection_conf = 0.0
            avg_knowledge_conf = 0.0

        final_confidence = self._confidence_scorer.score(
            evidence=all_evidence,
            detection_confidence=avg_detection_conf,
            knowledge_confidence=avg_knowledge_conf,
        )

        reason = self._build_reason(
            has_match,
            matched_object_count,
            all_evidence,
            final_confidence,
        )

        return FriendlyAnalysis(
            friendly_match=has_match,
            confidence=final_confidence,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_query(self, obj: DetectedObject) -> str:
        """Build a query string from a detected object's model class name."""
        return obj.name

    def _build_context(self, detection: DetectionResult) -> dict[str, Any]:
        """Build contextual metadata for the retrieval query."""
        return {
            "image_id": detection.image_id,
            "timestamp": detection.timestamp.isoformat(),
        }

    def _filter_relevant(
        self, items: list[KnowledgeItem], obj: DetectedObject
    ) -> list[KnowledgeItem]:
        """Filter knowledge items to those relevant for *obj*."""
        obj_type = obj.name.lower()

        relevant: list[KnowledgeItem] = []
        for item in items:
            if item.confidence < self._config.evidence_min_weight:
                continue
            if item.equipment:
                for equip in item.equipment:
                    if equip.lower() in obj_type:
                        relevant.append(item)
                        break
            if item not in relevant and item.markings:
                relevant.append(item)

        if not relevant:
            relevant = items[:self._config.max_knowledge_items]

        return relevant

    @staticmethod
    def _average_detection_confidence(detection: DetectionResult) -> float:
        """Return the mean detection confidence across all objects."""
        if not detection.objects:
            return 0.0
        total = sum(obj.confidence for obj in detection.objects)
        return total / len(detection.objects)

    @staticmethod
    def _build_reason(
        has_match: bool,
        count: int,
        evidence: list[Evidence],
        confidence: float,
    ) -> str:
        """Build a human-readable reason string."""
        if not has_match:
            return (
                f"No friendly-force match found (confidence={confidence:.2f}). "
                "No objects matched any known friendly unit in the knowledge base."
            )

        evidence_types = set(e.evidence_type for e in evidence)
        parts = ", ".join(sorted(evidence_types))
        return (
            f"Friendly match detected for {count} object(s) "
            f"(confidence={confidence:.2f}). "
            f"Matches based on: {parts}."
        )
