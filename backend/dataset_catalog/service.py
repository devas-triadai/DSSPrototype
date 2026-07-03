"""Dataset Catalog Service — public orchestrator for acquisition & curation.

This service sits upstream of Dataset Intelligence. It evaluates, scores,
prioritizes, and curates candidate datasets before they enter the import pipeline.

Pipeline:
  Discover → Profile → Score → Catalog → Coverage Analysis →
  Gap Analysis → Recommend → Plan Acquisition → Curate → Handoff to DI

All dependencies are injected. No concrete coupling.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_catalog.acquisition_planner import AcquisitionPlanner
from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.coverage_analyzer import CoverageAnalyzer
from backend.dataset_catalog.curation_service import CurationService
from backend.dataset_catalog.dataset_profile import DatasetProfiler
from backend.dataset_catalog.gap_analyzer import GapAnalyzer
from backend.dataset_catalog.interfaces import (
    AcquisitionPlannerInterface,
    CatalogInterface,
    CoverageAnalyzerInterface,
    CurationServiceInterface,
    DatasetCatalogServiceInterface,
    DatasetProfileInterface,
    GapAnalyzerInterface,
    LicenseManagerInterface,
    RecommendationEngineInterface,
    SourceRegistryInterface,
    TaxonomicalCoverageInterface,
)
from backend.dataset_catalog.license_manager import LicenseManager
from backend.dataset_catalog.models import (
    AcquisitionPlan,
    CatalogEntry,
    CoverageReport,
    CurationRecord,
    GapAnalysisReport,
    RecommendationResult,
)
from backend.dataset_catalog.recommendation_engine import RecommendationEngine
from backend.dataset_catalog.source_registry import SourceRegistry

logger = logging.getLogger("dss.dataset_catalog.service")


class DatasetCatalogService(DatasetCatalogServiceInterface):
    """Orchestrate the full Dataset Acquisition & Curation System.

    Every constructor parameter is optional and defaults to a production-grade
    implementation. This makes the service fully testable via dependency injection.
    """

    def __init__(
        self,
        catalog: CatalogInterface | None = None,
        source_registry: SourceRegistryInterface | None = None,
        profiler: DatasetProfileInterface | None = None,
        taxonomy: TaxonomicalCoverageInterface | None = None,
        license_manager: LicenseManagerInterface | None = None,
        coverage_analyzer: CoverageAnalyzerInterface | None = None,
        gap_analyzer: GapAnalyzerInterface | None = None,
        recommendation_engine: RecommendationEngineInterface | None = None,
        acquisition_planner: AcquisitionPlannerInterface | None = None,
        curation_service: CurationServiceInterface | None = None,
    ) -> None:
        self._catalog = catalog or Catalog()
        self._source_registry = source_registry or SourceRegistry()
        self._profiler = profiler or DatasetProfiler()
        self._taxonomy = taxonomy or ClassTaxonomy()
        self._license_manager = license_manager or LicenseManager()
        self._coverage_analyzer = coverage_analyzer or CoverageAnalyzer(
            self._catalog, self._taxonomy
        )
        self._gap_analyzer = gap_analyzer or GapAnalyzer(
            self._taxonomy, self._source_registry
        )
        self._recommendation_engine = recommendation_engine or RecommendationEngine(
            self._catalog, self._taxonomy, self._license_manager
        )
        self._acquisition_planner = acquisition_planner or AcquisitionPlanner(
            self._catalog
        )
        self._curation_service = curation_service or CurationService(
            self._catalog
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        path: Path,
        source_id: str,
        source_type: str,
        curator: str | None = None,
    ) -> CatalogEntry:
        """Discover and profile a candidate dataset, then add it to the catalog.

        This is the primary entry point for new datasets into the catalog.
        """
        logger.info(
            "Discovery started | path=%s | source=%s | type=%s",
            path, source_id, source_type,
        )

        # 1. Profile the dataset
        profile = self._profiler.profile(path, source_id, source_type)
        logger.info(
            "Profile complete | images=%d | annotations=%d | classes=%d",
            profile.total_images,
            profile.total_annotations,
            profile.total_classes,
        )

        # 2. Compute scores
        quality_score = self._compute_quality_score(profile)
        coverage = self._taxonomy.analyze_coverage(list(profile.classes))
        coverage_score = coverage.coverage_ratio
        diversity_score = self._compute_diversity_score(profile)
        license_score = 1.0
        if profile.license_info:
            risk = self._license_manager.compute_risk_score(profile.license_info)
            license_score = 1.0 - risk

        overall_score = (
            dc_config.weight_quality * quality_score
            + dc_config.weight_coverage * coverage_score
            + dc_config.weight_diversity * diversity_score
            + dc_config.weight_license * license_score
            + dc_config.weight_source_reliability * 0.5
        )

        # 3. Create catalog entry
        entry_id = profile.profile_id
        estimated_budget = round(overall_score * 1000, 2)
        estimated_storage = round(profile.estimated_size_mb, 2)
        entry = CatalogEntry(
            entry_id=entry_id,
            name=path.name,
            source_id=source_id,
            source_type=source_type,
            domain="military",
            status="profiled",
            profile=profile,
            quality_score=round(quality_score, 4),
            coverage_score=round(coverage_score, 4),
            diversity_score=round(diversity_score, 4),
            license_score=round(license_score, 4),
            overall_score=round(overall_score, 4),
            estimated_budget=estimated_budget,
            estimated_storage_mb=estimated_storage,
            tags=profile.tags,
        )

        self._catalog.add_entry(entry)
        logger.info(
            "Catalog entry created | entry=%s | overall=%.4f",
            entry.entry_id, entry.overall_score,
        )

        # 4. Auto-create curation record if curator specified
        if curator:
            try:
                record = self._curation_service.create_record(entry.entry_id, curator)
                if dc_config.auto_curation_enabled:
                    self._curation_service.submit_for_review(record.record_id)
                logger.info(
                    "Curation record created | record=%s | curator=%s",
                    record.record_id, curator,
                )
            except Exception as exc:
                logger.warning("Curation record creation failed: %s", exc)

        logger.info("Discovery complete | entry=%s", entry.entry_id)
        return entry

    def get_catalog_coverage(self) -> CoverageReport:
        """Get a holistic coverage report across the entire catalog."""
        logger.info("Coverage analysis started")
        report = self._coverage_analyzer.analyze_all()
        logger.info(
            "Coverage analysis complete | entries=%d | aggregate=%.4f",
            report.total_entries_analyzed,
            report.aggregate_coverage,
        )
        return report

    def get_gap_analysis(self) -> GapAnalysisReport:
        """Get a gap analysis report identifying coverage deficiencies."""
        logger.info("Gap analysis started")
        coverage = self.get_catalog_coverage()
        report = self._gap_analyzer.identify_gaps(coverage)
        logger.info(
            "Gap analysis complete | gaps=%d | critical=%d | severity=%.4f",
            report.total_gap_count,
            report.critical_gap_count,
            report.aggregate_gap_severity,
        )
        return report

    def get_recommendations(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[RecommendationResult]:
        """Get the top dataset recommendations."""
        logger.info("Recommendations requested | domain=%s | limit=%d", domain, limit)
        results = self._recommendation_engine.recommend(domain, limit)
        logger.info("Recommendations generated | count=%d", len(results))
        return results

    def recommend_for_gap(
        self, gap_node_id: str
    ) -> list[RecommendationResult]:
        """Get recommendations to fill a specific taxonomy gap."""
        logger.info("Gap recommendations requested | gap=%s", gap_node_id)
        results = self._recommendation_engine.recommend_for_gap(gap_node_id)
        logger.info("Gap recommendations generated | count=%d", len(results))
        return results

    def create_acquisition_plan(
        self,
        entries: Sequence[str],
        priority: float,
        notes: str = "",
    ) -> AcquisitionPlan:
        """Create an acquisition plan for one or more catalog entries."""
        logger.info(
            "Acquisition plan creation | entries=%s | priority=%.2f",
            entries, priority,
        )

        # Look up entries for budget/target domains
        estimated_budget = 0.0
        estimated_storage = 0.0
        target_domains: list[str] = []
        for eid in entries:
            entry = self._catalog.get_entry(eid)
            if entry and entry.profile:
                estimated_budget += entry.overall_score * 1000
                estimated_storage += entry.profile.estimated_size_mb
                if entry.domain not in target_domains:
                    target_domains.append(entry.domain)

        plan = AcquisitionPlan(
            plan_id=f"plan_{'_'.join(entries)}",
            entry_ids=list(entries),
            priority=priority,
            status="draft",
            estimated_budget=round(estimated_budget, 2),
            estimated_storage_mb=round(estimated_storage, 2),
            target_domains=target_domains,
            notes=notes,
        )

        result = self._acquisition_planner.create_plan(plan)
        logger.info("Acquisition plan created | plan=%s", result.plan_id)
        return result

    def submit_for_curation(
        self, entry_id: str, curator: str
    ) -> CurationRecord:
        """Submit a catalog entry for curation review."""
        logger.info("Curation submission | entry=%s | curator=%s", entry_id, curator)
        record = self._curation_service.create_record(entry_id, curator)
        record = self._curation_service.submit_for_review(record.record_id)
        logger.info("Curation submitted | record=%s | status=pending_review", record.record_id)
        return record

    def approve_curation(
        self, record_id: str, reviewer: str
    ) -> CurationRecord:
        """Approve a pending curation record."""
        logger.info("Curation approval | record=%s | reviewer=%s", record_id, reviewer)
        record = self._curation_service.approve(record_id, reviewer)
        logger.info("Curation approved | record=%s | entry=%s", record_id, record.entry_id)
        return record

    def reject_curation(
        self, record_id: str, reviewer: str, reason: str
    ) -> CurationRecord:
        """Reject a pending curation record."""
        logger.info("Curation rejection | record=%s | reviewer=%s", record_id, reviewer)
        record = self._curation_service.reject(record_id, reviewer, reason)
        logger.info("Curation rejected | record=%s", record_id)
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_score(profile: object) -> float:
        """Compute a quality score from a dataset profile."""
        from backend.dataset_catalog.models import DatasetProfile

        if not isinstance(profile, DatasetProfile):
            return 0.0

        score = profile.format_consistency
        if profile.total_images > 0:
            missing_penalty = 1.0 - min(
                profile.missing_annotations / profile.total_images, 0.5
            )
            corrupt_penalty = 1.0 - min(
                profile.corrupt_images / profile.total_images, 0.5
            )
            score *= missing_penalty * corrupt_penalty
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _compute_diversity_score(profile: object) -> float:
        """Compute a diversity score based on class distribution uniformity."""
        from backend.dataset_catalog.models import DatasetProfile

        if not isinstance(profile, DatasetProfile):
            return 0.0

        if not profile.class_distribution:
            return 0.0

        counts = [cd.count for cd in profile.class_distribution]
        if not counts:
            return 0.0

        max_c = max(counts)
        min_c = min(counts)
        if max_c > 0:
            return 1.0 - (max_c - min_c) / max_c
        return 0.0
