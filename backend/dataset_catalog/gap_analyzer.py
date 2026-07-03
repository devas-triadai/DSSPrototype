"""Gap analysis — identifies coverage deficiencies in the taxonomy."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.interfaces import (
    GapAnalyzerInterface,
    SourceRegistryInterface,
    TaxonomicalCoverageInterface,
)
from backend.dataset_catalog.models import (
    CoverageReport,
    GapAnalysisReport,
    GapEntry,
    TaxonomyNode,
)

logger = logging.getLogger("dss.dataset_catalog.gap_analyzer")


class GapAnalyzer(GapAnalyzerInterface):
    """Identifies and prioritizes gaps in dataset coverage against the taxonomy."""

    def __init__(
        self,
        taxonomy: TaxonomicalCoverageInterface,
        source_registry: SourceRegistryInterface | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self._taxonomy = taxonomy
        self._source_registry = source_registry
        self._reports_dir = reports_dir or dc_config.reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def identify_gaps(
        self,
        coverage: CoverageReport,
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> GapAnalysisReport:
        """Identify gaps in coverage against the taxonomy.

        A gap is defined as a taxonomy node that is not fully covered.
        Severity is calculated based on:
          - Node priority (higher priority = higher severity)
          - Whether the node is completely uncovered vs partially covered
          - Domain-level weakness
        """
        taxonomy_nodes = taxonomy if taxonomy is not None else self._taxonomy.load_taxonomy()
        nodes = taxonomy_nodes

        # Aggregate covered node IDs across all entries
        covered_set: set[str] = set()
        partial_set: set[str] = set()
        for entry_cov in coverage.entries:
            covered_set.update(entry_cov.covered_nodes)
            partial_set.update(entry_cov.partial_nodes)

        gaps: list[GapEntry] = []
        for node in nodes:
            if node.node_id in covered_set:
                continue
            is_partial = node.node_id in partial_set

            # Calculate severity
            domain_coverage = coverage.domain_breakdown.get(node.domain, 0.0)
            base_severity = 1.0 - domain_coverage
            priority_factor = node.priority
            partial_penalty = 0.5 if is_partial else 1.0

            severity = min(base_severity * priority_factor * partial_penalty * 1.5, 1.0)

            if severity >= 0.8:
                impact = "critical"
            elif severity >= 0.6:
                impact = "high"
            elif severity >= 0.4:
                impact = "medium"
            else:
                impact = "low"

            potential_sources = self._find_potential_sources(node)

            gaps.append(
                GapEntry(
                    node_id=node.node_id,
                    node_name=node.name,
                    domain=node.domain,
                    severity=round(severity, 4),
                    description=(
                        f"{'Partial' if is_partial else 'No'} coverage for {node.name} "
                        f"(domain: {node.domain}, priority: {node.priority})"
                    ),
                    potential_sources=potential_sources,
                    impact=impact,
                )
            )

        critical = sum(1 for g in gaps if g.impact == "critical")
        high = sum(1 for g in gaps if g.impact == "high")
        medium = sum(1 for g in gaps if g.impact == "medium")
        low = sum(1 for g in gaps if g.impact == "low")

        domain_breakdown: dict[str, int] = {}
        for g in gaps:
            domain_breakdown[g.domain] = domain_breakdown.get(g.domain, 0) + 1

        aggregate_severity = (
            sum(g.severity for g in gaps) / len(gaps) if gaps else 0.0
        )

        report = GapAnalysisReport(
            report_id=f"gap_{datetime.now(timezone.utc).timestamp()}",
            gaps=gaps,
            critical_gap_count=critical,
            high_gap_count=high,
            medium_gap_count=medium,
            low_gap_count=low,
            total_gap_count=len(gaps),
            aggregate_gap_severity=round(aggregate_severity, 4),
            domain_breakdown=domain_breakdown,
        )

        self._persist(report)
        return report

    def get_critical_gaps(self, report: GapAnalysisReport) -> list[str]:
        return [g.node_id for g in report.gaps if g.impact == "critical"]

    def get_gap_recommendations(
        self, report: GapAnalysisReport
    ) -> list[dict[str, object]]:
        recommendations: list[dict[str, object]] = []
        critical = [g for g in report.gaps if g.impact == "critical"]
        high = [g for g in report.gaps if g.impact == "high"]

        for gap in critical[:5]:
            recommendations.append(
                {
                    "node_id": gap.node_id,
                    "node_name": gap.node_name,
                    "priority": "critical",
                    "action": "Acquire datasets covering this node immediately",
                    "potential_sources": gap.potential_sources,
                    "reason": f"Highest severity gap ({gap.severity:.2f}) in domain '{gap.domain}'",
                }
            )

        for gap in high[:5]:
            recommendations.append(
                {
                    "node_id": gap.node_id,
                    "node_name": gap.node_name,
                    "priority": "high",
                    "action": "Prioritize acquisition for this taxonomy node",
                    "potential_sources": gap.potential_sources,
                    "reason": f"High severity gap ({gap.severity:.2f}) in domain '{gap.domain}'",
                }
            )

        return recommendations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_potential_sources(self, node: TaxonomyNode) -> list[str]:
        """Find potential sources that might cover this gap.

        Uses keyword matching against registered source descriptions.
        Returns source IDs of matching sources.
        """
        if self._source_registry is None:
            return []
        sources = self._source_registry.list_sources()
        keywords = set(k.lower() for k in node.keywords)
        keyword_parts: set[str] = set()
        for kw in keywords:
            keyword_parts.update(kw.replace("_", " ").replace("-", " ").split())

        matches: list[str] = []
        for src in sources:
            desc_lower = src.description.lower()
            name_lower = src.name.lower()
            if any(kw in desc_lower or kw in name_lower for kw in keyword_parts):
                matches.append(src.source_id)
        return matches[:5]

    def _persist(self, report: GapAnalysisReport) -> None:
        path = self._reports_dir / f"{report.report_id}.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
