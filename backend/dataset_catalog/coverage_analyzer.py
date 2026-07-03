"""Holistic coverage analysis across the dataset catalog."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.interfaces import (
    CatalogInterface,
    CoverageAnalyzerInterface,
    TaxonomicalCoverageInterface,
)
from backend.dataset_catalog.models import (
    CoverageReport,
    DatasetProfile,
    TaxonomicalCoverage,
)

logger = logging.getLogger("dss.dataset_catalog.coverage_analyzer")


class CoverageAnalyzer(CoverageAnalyzerInterface):
    """Analyzes coverage of the taxonomy across catalog entries."""

    def __init__(
        self,
        catalog: CatalogInterface,
        taxonomy: TaxonomicalCoverageInterface,
        reports_dir: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._taxonomy = taxonomy
        self._reports_dir = reports_dir or dc_config.reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_all(self) -> CoverageReport:
        """Analyze coverage across all catalog entries with non-zero profiles."""
        entries = self._catalog.list_entries()
        profiles = [e.profile for e in entries if e.profile is not None]
        return self._analyze_profiles(profiles)

    def analyze_domain(self, domain: str) -> CoverageReport:
        entries = self._catalog.list_entries(domain=domain)
        profiles = [e.profile for e in entries if e.profile is not None]
        return self._analyze_profiles(profiles)

    def analyze_entries(self, entry_ids: Sequence[str]) -> CoverageReport:
        profiles: list[DatasetProfile] = []
        for eid in entry_ids:
            entry = self._catalog.get_entry(eid)
            if entry and entry.profile:
                profiles.append(entry.profile)
        return self._analyze_profiles(profiles)

    def coverage_trend(self, days: int = 30) -> list[dict[str, object]]:
        """Return coverage metrics over time (stub — requires historical data)."""
        return [
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "coverage": 0.0,
                "entries_analyzed": 0,
            }
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_profiles(self, profiles: list[DatasetProfile]) -> CoverageReport:
        if not profiles:
            return CoverageReport(
                report_id=f"cr_{datetime.now(timezone.utc).timestamp()}",
                total_entries_analyzed=0,
                aggregate_coverage=0.0,
            )

        coverages: list[TaxonomicalCoverage] = []
        for profile in profiles:
            cov = self._taxonomy.analyze_coverage(list(profile.classes))
            coverages.append(cov)

        total_coverage = (
            sum(c.coverage_ratio for c in coverages) / len(coverages)
            if coverages
            else 0.0
        )

        domain_lists: dict[str, list[float]] = {}
        for cov in coverages:
            for domain, ratio in cov.domain_coverage.items():
                if domain not in domain_lists:
                    domain_lists[domain] = []
                domain_lists[domain].append(ratio)

        avg_domain: dict[str, float] = {}
        for domain, ratios in domain_lists.items():
            avg_domain[domain] = sum(ratios) / len(ratios)

        sorted_domains = sorted(avg_domain.items(), key=lambda x: x[1])
        weakest = [d for d, _ in sorted_domains[:3]]
        strongest = [d for d, _ in sorted_domains[-3:]]

        report = CoverageReport(
            report_id=f"cr_{datetime.now(timezone.utc).timestamp()}",
            total_entries_analyzed=len(profiles),
            entries=coverages,
            aggregate_coverage=total_coverage,
            domain_breakdown=avg_domain,
            weakest_domains=weakest,
            strongest_domains=strongest,
        )

        self._persist(report)
        return report

    def _persist(self, report: CoverageReport) -> None:
        path = self._reports_dir / f"{report.report_id}.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
