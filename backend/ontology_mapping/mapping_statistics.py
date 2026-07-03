"""Statistics generation and reporting for ontology mappings.

Computes coverage, conflict counts, compatibility scores,
and generates human-readable reports.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from backend.ontology_mapping.models import (
    MappingRule,
    MappingStatistics,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver


class MappingStatisticsGenerator:
    """Generates statistics and reports for dataset ontology mappings."""

    def __init__(
        self,
        resolver: OntologyResolver | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()

    async def compute(
        self,
        dataset_name: str,
        rules: list[MappingRule],
    ) -> MappingStatistics:
        all_labels: set[str] = set()
        mapped: set[str] = set()
        ignored: set[str] = set()
        unknown: set[str] = set()

        for rule in rules:
            label_lower = rule.source_label.lower()
            all_labels.add(label_lower)
            exists = await self._resolver.contains(rule.canonical_value)
            if not exists:
                unknown.add(label_lower)
            elif rule.canonical_value == "unknown_object":
                unknown.add(label_lower)
            else:
                mapped.add(label_lower)

        duplicated_aliases = self._count_duplicate_aliases(rules)
        total = len(all_labels) or 1
        coverage = (len(mapped) / total) * 100.0

        conflict_count = self._count_conflicts(rules)

        return MappingStatistics(
            dataset_name=dataset_name,
            total_labels=len(all_labels),
            mapped_labels=len(mapped),
            ignored_labels=len(ignored),
            unknown_labels=len(unknown),
            coverage_percent=round(coverage, 2),
            conflict_count=conflict_count,
            duplicate_alias_count=duplicated_aliases,
            ontology_version="1.0.0",
            generated_at=datetime.now(timezone.utc),
        )

    async def coverage_report(
        self,
        statistics: MappingStatistics,
    ) -> str:
        lines: list[str] = [
            f"Dataset: {statistics.dataset_name}",
            f"Generated: {statistics.generated_at.isoformat()}",
            f"Ontology version: {statistics.ontology_version}",
            "",
            "--- Coverage ---",
            f"Total labels:     {statistics.total_labels}",
            f"Mapped labels:    {statistics.mapped_labels}",
            f"Ignored labels:   {statistics.ignored_labels}",
            f"Unknown labels:   {statistics.unknown_labels}",
            f"Coverage:         {statistics.coverage_percent}%",
            "",
            "--- Health ---",
            f"Conflicts:        {statistics.conflict_count}",
            f"Duplicate aliases:{statistics.duplicate_alias_count}",
            "",
        ]

        if statistics.coverage_percent >= 95.0:
            lines.append("Status: HEALTHY (coverage >= 95%)")
        elif statistics.coverage_percent >= 80.0:
            lines.append("Status: ACCEPTABLE (coverage >= 80%)")
        else:
            lines.append("Status: NEEDS_ATTENTION (coverage < 80%)")

        return "\n".join(lines)

    async def compatibility_report(
        self,
        source: MappingStatistics,
        target: MappingStatistics,
    ) -> str:
        coverage_diff = abs(
            source.coverage_percent - target.coverage_percent
        )
        lines: list[str] = [
            "--- Compatibility Report ---",
            f"Dataset A: {source.dataset_name} (coverage {source.coverage_percent}%)",
            f"Dataset B: {target.dataset_name} (coverage {target.coverage_percent}%)",
            f"Coverage difference: {coverage_diff:.2f}%",
            f"A conflicts: {source.conflict_count}",
            f"B conflicts: {target.conflict_count}",
            "",
        ]
        if coverage_diff <= 10.0:
            lines.append("Verdict: COMPATIBLE (coverage diff <= 10%)")
        else:
            lines.append("Verdict: INCOMPATIBLE (coverage diff > 10%)")
        return "\n".join(lines)

    def _count_duplicate_aliases(
        self,
        rules: list[MappingRule],
    ) -> int:
        alias_map: dict[str, set[str]] = defaultdict(set)
        for rule in rules:
            alias_map[rule.source_label.lower()].add(rule.canonical_value)
        return sum(
            len(values) - 1
            for values in alias_map.values()
            if len(values) > 1
        )

    def _count_conflicts(
        self,
        rules: list[MappingRule],
    ) -> int:
        label_to_values: dict[str, set[str]] = defaultdict(set)
        for rule in rules:
            label_to_values[rule.source_label.lower()].add(
                rule.canonical_value
            )
        return sum(
            1 for v in label_to_values.values() if len(v) > 1
        )
