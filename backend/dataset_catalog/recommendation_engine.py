"""Recommendation engine — scores and recommends datasets from the catalog."""

from __future__ import annotations

import logging

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import EntryNotFoundError
from backend.dataset_catalog.interfaces import (
    CatalogInterface,
    LicenseManagerInterface,
    RecommendationEngineInterface,
    TaxonomicalCoverageInterface,
)
from backend.dataset_catalog.models import (
    CatalogEntry,
    RecommendationResult,
)

logger = logging.getLogger("dss.dataset_catalog.recommendation_engine")


class RecommendationEngine(RecommendationEngineInterface):
    """Scores catalog entries and generates dataset recommendations.

    Scoring dimensions:
      - Quality score (from profile quality indicators)
      - Coverage score (taxonomy coverage ratio)
      - Diversity score (class distribution uniformity)
      - License score (inverse of license risk)
      - Overall score (weighted combination)
    """

    def __init__(
        self,
        catalog: CatalogInterface,
        taxonomy: TaxonomicalCoverageInterface,
        license_manager: LicenseManagerInterface,
    ) -> None:
        self._catalog = catalog
        self._taxonomy = taxonomy
        self._license_manager = license_manager

    def score_entry(self, entry_id: str) -> RecommendationResult:
        """Score a single catalog entry."""
        entry = self._catalog.get_entry(entry_id)
        if entry is None:
            raise EntryNotFoundError(f"Entry not found: {entry_id}")

        return self._score(entry)

    def recommend(
        self,
        domain: str | None = None,
        limit: int = 20,
        min_score: float = 0.3,
    ) -> list[RecommendationResult]:
        """Return the top-N dataset recommendations."""
        entries = self._catalog.list_entries()
        if domain:
            entries = [e for e in entries if e.domain == domain]

        scored = [self._score(e) for e in entries]
        scored = [s for s in scored if s.overall_score >= min_score]
        scored.sort(key=lambda s: s.overall_score, reverse=True)
        return scored[:limit]

    def recommend_for_gap(
        self,
        gap_node_id: str,
        limit: int = 10,
    ) -> list[RecommendationResult]:
        """Recommend datasets that best fill a specific taxonomy gap."""
        entries = self._catalog.list_entries()
        scored: list[RecommendationResult] = []

        for entry in entries:
            result = self._score(entry)
            # Check if this entry's classes relate to the gap node
            if entry.profile:
                coverage = self._taxonomy.analyze_coverage(
                    list(entry.profile.classes)
                )
                if gap_node_id in coverage.covered_nodes:
                    scored.append(result)

        scored.sort(key=lambda s: s.overall_score, reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, entry: CatalogEntry) -> RecommendationResult:
        profile = entry.profile

        quality_score: float
        coverage_score: float
        diversity_score: float
        license_score: float

        if profile:
            quality = profile.format_consistency
            max_img = max(profile.total_images, 1)
            missing_penalty = 1.0 - min(profile.missing_annotations / max_img, 0.5)
            corrupt_penalty = 1.0 - min(profile.corrupt_images / max_img, 0.5)
            quality_score = quality * missing_penalty * corrupt_penalty
            quality_score = quality * missing_penalty * corrupt_penalty
        else:
            quality_score = 0.0

        if profile and profile.classes:
            coverage = self._taxonomy.analyze_coverage(list(profile.classes))
            coverage_score = coverage.coverage_ratio
        else:
            coverage_score = 0.0

        if profile and profile.class_distribution:
            counts = [cd.count for cd in profile.class_distribution]
            if counts:
                max_c = max(counts)
                min_c = min(counts)
                if max_c > 0:
                    diversity_score = 1.0 - (max_c - min_c) / max_c
                else:
                    diversity_score = 0.0
            else:
                diversity_score = 0.0
        else:
            diversity_score = 0.0

        if entry.license_score > 0:
            license_score = entry.license_score
        elif profile and profile.license_info:
            risk = self._license_manager.compute_risk_score(profile.license_info)
            license_score = 1.0 - risk
        else:
            license_score = 0.5

        # Overall weighted score
        overall = (
            dc_config.weight_quality * quality_score
            + dc_config.weight_coverage * coverage_score
            + dc_config.weight_diversity * diversity_score
            + dc_config.weight_license * license_score
            + dc_config.weight_source_reliability * entry.quality_score
        )

        reason_parts: list[str] = []
        if quality_score >= 0.7:
            reason_parts.append("high quality")
        if coverage_score >= 0.6:
            reason_parts.append(f"good taxonomy coverage ({coverage_score:.0%})")
        if diversity_score >= 0.6:
            reason_parts.append("diverse class distribution")
        if license_score >= 0.8:
            reason_parts.append("permissive license")

        reason = "; ".join(reason_parts) if reason_parts else "moderate scores"

        return RecommendationResult(
            entry_id=entry.entry_id,
            entry_name=entry.name,
            domain=entry.domain,
            quality_score=round(quality_score, 4),
            coverage_score=round(coverage_score, 4),
            diversity_score=round(diversity_score, 4),
            license_score=round(license_score, 4),
            overall_score=round(overall, 4),
            reason=reason,
        )
