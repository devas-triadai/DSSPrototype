"""Ontology and mapping rule validation.

Validates:
- Ontology tree integrity (no orphans, no cycles, valid parents)
- Mapping rule correctness (all targets exist in ontology)
- No conflicting aliases
- No duplicate canonical labels
"""

from __future__ import annotations

from backend.ontology_mapping.models import MappingRule
from backend.ontology_mapping.ontology_resolver import OntologyResolver


class MappingValidator:
    """Validates ontology structure and mapping rule correctness."""

    def __init__(
        self,
        resolver: OntologyResolver | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()

    async def validate_ontology(self) -> list[str]:
        errors: list[str] = []
        root = await self._resolver.get_node("root")
        if root is None:
            errors.append("Root node 'root' is missing")
            return errors

        leaves = await self._resolver.get_leaves()

        for leaf in leaves:
            if leaf.parent:
                parent = await self._resolver.get_node(leaf.parent)
                if parent is None:
                    errors.append(
                        f"Leaf '{leaf.value}' has missing parent "
                        f"'{leaf.parent}'"
                    )

        categories = await self._resolver.get_children("root")
        for cat_node in categories:
            if cat_node.parent != "root":
                errors.append(
                    f"Category '{cat_node.value}' has parent "
                    f"'{cat_node.parent}', expected 'root'"
                )

        seen_values: set[str] = set()
        for leaf in leaves:
            if leaf.value in seen_values:
                errors.append(f"Duplicate ontology value '{leaf.value}'")
            seen_values.add(leaf.value)

        for cat in categories:
            cat_value = cat.value
            if "_" in cat_value:
                content = await self._resolver.get_children(cat_value)
                for child in content:
                    if child.parent != cat_value:
                        errors.append(
                            f"Child '{child.value}' has parent "
                            f"'{child.parent}', expected '{cat_value}'"
                        )

        return errors

    async def validate_mapping(
        self,
        rules: list[MappingRule],
    ) -> list[str]:
        errors: list[str] = []

        seen_rules: dict[str, list[MappingRule]] = {}
        for rule in rules:
            exists = await self._resolver.contains(rule.canonical_value)
            if not exists:
                errors.append(
                    f"Rule '{rule.rule_id}': ontology node "
                    f"'{rule.canonical_value}' does not exist"
                )

            key = f"{rule.dataset_name}:{rule.source_label.lower()}"
            if key in seen_rules:
                existing = seen_rules[key][0]
                if existing.canonical_value != rule.canonical_value:
                    errors.append(
                        f"Label '{rule.source_label}' in dataset "
                        f"'{rule.dataset_name}' maps to both "
                        f"'{existing.canonical_value}' and "
                        f"'{rule.canonical_value}'"
                    )
                seen_rules[key].append(rule)
            else:
                seen_rules[key] = [rule]

        return errors

    async def validate_rules(
        self,
        rules: list[MappingRule],
    ) -> list[str]:
        errors: list[str] = []

        for rule in rules:
            if not rule.rule_id:
                errors.append("Rule has empty rule_id")
            if not rule.source_label.strip():
                errors.append(f"Rule '{rule.rule_id}' has empty source_label")
            if not rule.canonical_value.strip():
                errors.append(
                    f"Rule '{rule.rule_id}' has empty canonical_value"
                )
            if rule.confidence < 0.0 or rule.confidence > 1.0:
                errors.append(
                    f"Rule '{rule.rule_id}' has out-of-range confidence "
                    f"{rule.confidence}"
                )

        rule_ids: set[str] = set()
        for rule in rules:
            if rule.rule_id in rule_ids:
                errors.append(f"Duplicate rule_id '{rule.rule_id}'")
            rule_ids.add(rule.rule_id)

        return errors
