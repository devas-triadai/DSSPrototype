"""Conflict detection and resolution for ontology mapping rules.

Conflicts are detected by analysing the registry for:
- Duplicate mappings (same rule twice)
- Conflicting mappings (same label → different ontology nodes)
- Circular mappings (indirectly via alias chains)
- Missing ontology nodes
- Ambiguous aliases (alias matches multiple nodes)
"""

from __future__ import annotations

from collections import defaultdict

from backend.ontology_mapping.models import (
    ConflictResolution,
    ConflictType,
    MappingConflict,
    MappingRule,
    ResolutionType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver


class ConflictResolver:
    """Detects and deterministically resolves mapping conflicts."""

    def __init__(
        self,
        resolver: OntologyResolver | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()

    async def detect(
        self,
        rules: list[MappingRule],
    ) -> list[MappingConflict]:
        conflicts: list[MappingConflict] = []

        # 1. Detect exact duplicates
        seen_ids: set[str] = set()
        for rule in rules:
            if rule.rule_id in seen_ids:
                conflicts.append(
                    MappingConflict(
                        conflict_type=ConflictType.DUPLICATE,
                        source_label=rule.source_label,
                        rules=(rule, rule),
                        description=(
                            f"Duplicate rule '{rule.rule_id}' for "
                            f"label '{rule.source_label}'"
                        ),
                    )
                )
            seen_ids.add(rule.rule_id)

        # 2. Detect conflicting mappings (same label → different values)
        label_to_rules: dict[str, list[MappingRule]] = defaultdict(list)
        for rule in rules:
            label_to_rules[rule.source_label.lower()].append(rule)
        for label_lower, label_rules in label_to_rules.items():
            unique_values = {r.canonical_value for r in label_rules}
            if len(unique_values) > 1:
                conflicts.append(
                    MappingConflict(
                        conflict_type=ConflictType.CONFLICTING,
                        source_label=label_rules[0].source_label,
                        rules=tuple(sorted(
                            label_rules, key=lambda r: r.created_at
                        )),
                        description=(
                            f"Label '{label_rules[0].source_label}' maps to "
                            f"{len(unique_values)} different ontology values"
                        ),
                    )
                )

        # 3. Detect missing ontology nodes
        for rule in rules:
            exists = await self._resolver.contains(rule.canonical_value)
            if not exists:
                conflicts.append(
                    MappingConflict(
                        conflict_type=ConflictType.MISSING_NODE,
                        source_label=rule.source_label,
                        rules=(rule,),
                        description=(
                            f"Ontology node '{rule.canonical_value}' does not exist"
                        ),
                    )
                )

        # 4. Detect circular alias chains
        alias_graph: dict[str, set[str]] = defaultdict(set)
        for rule in rules:
            if rule.match_type.value in ("alias", "synonym"):
                alias_graph[rule.canonical_value].add(rule.source_label.lower())

        for start in alias_graph:
            visited: set[str] = set()
            stack = [start]
            while stack:
                current = stack.pop()
                if current in visited:
                    conflicts.append(
                        MappingConflict(
                            conflict_type=ConflictType.CIRCULAR,
                            source_label=current,
                            rules=(),
                            description=(
                                f"Circular alias chain detected involving "
                                f"'{current}'"
                            ),
                        )
                    )
                    break
                visited.add(current)
                for neighbor in alias_graph.get(current, set()):
                    if neighbor in alias_graph:
                        stack.append(neighbor)

        # 5. Detect ambiguous aliases (alias matching multiple nodes)
        alias_to_values: dict[str, set[str]] = defaultdict(set)
        for rule in rules:
            if rule.match_type in (
                "alias", "synonym",
            ) or rule.match_type.value in ("alias", "synonym"):
                key = rule.source_label.lower()
                alias_to_values[key].add(rule.canonical_value)
        for alias, values in alias_to_values.items():
            if len(values) > 1:
                matching_rules = [
                    r for r in rules if r.source_label.lower() == alias
                ]
                if matching_rules:
                    conflicts.append(
                        MappingConflict(
                            conflict_type=ConflictType.AMBIGUOUS_ALIAS,
                            source_label=matching_rules[0].source_label,
                            rules=tuple(sorted(
                                matching_rules, key=lambda r: r.created_at
                            )),
                            description=(
                                f"Alias '{alias}' maps to "
                                f"{len(values)} different ontology values: "
                                f"{', '.join(sorted(values))}"
                            ),
                        )
                    )

        return conflicts

    async def resolve(
        self,
        conflict: MappingConflict,
        strategy: ResolutionType = ResolutionType.HIGHEST_CONFIDENCE,
    ) -> ConflictResolution:
        if conflict.conflict_type == ConflictType.DUPLICATE:
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=ResolutionType.FIRST_WINS,
                chosen_rule=conflict.rules[0],
                explanation="Duplicate removed, keeping first occurrence",
            )

        if conflict.conflict_type == ConflictType.CONFLICTING:
            return self._resolve_conflicting(
                conflict, strategy
            )

        if conflict.conflict_type == ConflictType.CIRCULAR:
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=ResolutionType.MANUAL,
                explanation="Circular dependencies require manual resolution",
            )

        if conflict.conflict_type == ConflictType.MISSING_NODE:
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=ResolutionType.MANUAL,
                explanation=(
                    "Missing ontology node must be resolved by updating "
                    "the ontology or removing the rule"
                ),
            )

        if conflict.conflict_type == ConflictType.AMBIGUOUS_ALIAS:
            return self._resolve_conflicting(
                conflict, strategy
            )

        return ConflictResolution(
            conflict_id=conflict.conflict_id,
            resolution_type=ResolutionType.MANUAL,
            explanation="Unrecognised conflict type requires manual resolution",
        )

    async def resolve_all(
        self,
        conflicts: list[MappingConflict],
    ) -> list[ConflictResolution]:
        return [
            await self.resolve(c) for c in conflicts
        ]

    def _resolve_conflicting(
        self,
        conflict: MappingConflict,
        strategy: ResolutionType,
    ) -> ConflictResolution:
        if not conflict.rules:
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=ResolutionType.MANUAL,
                explanation="No rules to resolve",
            )

        if strategy == ResolutionType.FIRST_WINS:
            chosen = conflict.rules[0]
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=strategy,
                chosen_rule=chosen,
                explanation=(
                    f"Selected first rule '{chosen.rule_id}' mapping "
                    f"'{chosen.source_label}' → '{chosen.canonical_value}'"
                ),
            )

        if strategy == ResolutionType.LAST_WINS:
            chosen = conflict.rules[-1]
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=strategy,
                chosen_rule=chosen,
                explanation=(
                    f"Selected last rule '{chosen.rule_id}' mapping "
                    f"'{chosen.source_label}' → '{chosen.canonical_value}'"
                ),
            )

        if strategy == ResolutionType.HIGHEST_CONFIDENCE:
            chosen = max(conflict.rules, key=lambda r: r.confidence)
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=strategy,
                chosen_rule=chosen,
                explanation=(
                    f"Selected highest-confidence rule '{chosen.rule_id}' "
                    f"(confidence={chosen.confidence}) mapping "
                    f"'{chosen.source_label}' → '{chosen.canonical_value}'"
                ),
            )

        if strategy == ResolutionType.MERGE:
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                resolution_type=strategy,
                merged_value=conflict.rules[0].canonical_value,
                explanation="Merge not applicable for conflicting mappings",
            )

        return ConflictResolution(
            conflict_id=conflict.conflict_id,
            resolution_type=ResolutionType.MANUAL,
            explanation="No automatic resolution strategy applies",
        )
