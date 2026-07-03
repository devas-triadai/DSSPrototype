"""Label-to-ontology translation engine.

Supports multiple matching strategies with configurable confidence scoring.
"""

from __future__ import annotations

import re

from backend.ontology_mapping.models import (
    MappingResult,
    MappingRule,
    MatchType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver


class MappingEngine:
    """Translates dataset labels to canonical ontology values.

    Matching is attempted in order of precision:
    1. Exact match
    2. Case-insensitive match
    3. Alias match (from known alias tables)
    4. Plural/singular normalization
    5. Synonym match (from synonym tables)
    6. Regex pattern match
    """

    def __init__(
        self,
        resolver: OntologyResolver | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()
        self._aliases: dict[str, set[str]] = {}
        self._synonyms: dict[str, set[str]] = {}
        self._regex_rules: list[tuple[re.Pattern[str], str, float]] = []

    def register_alias(
        self,
        canonical_value: str,
        alias: str,
    ) -> None:
        if canonical_value not in self._aliases:
            self._aliases[canonical_value] = set()
        self._aliases[canonical_value].add(alias.lower())

    def register_synonym(
        self,
        canonical_value: str,
        synonym: str,
    ) -> None:
        if canonical_value not in self._synonyms:
            self._synonyms[canonical_value] = set()
        self._synonyms[canonical_value].add(synonym.lower())

    def register_regex(
        self,
        pattern: str,
        canonical_value: str,
        confidence: float = 0.7,
    ) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)
        self._regex_rules.append((compiled, canonical_value, confidence))

    async def map_label(
        self,
        dataset_name: str,
        label: str,
        rules: list[MappingRule],
    ) -> MappingResult:
        label_stripped = label.strip()
        label_lower = label_stripped.lower()

        # 1. Check registry rules first
        for rule in rules:
            if rule.source_label.strip() == label_stripped:
                node = await self._resolver.get_node(rule.canonical_value)
                return MappingResult(
                    source_label=label_stripped,
                    canonical_value=rule.canonical_value,
                    canonical_name=node.name if node else rule.canonical_value,
                    confidence=rule.confidence,
                    match_type=MatchType.EXACT,
                    rule_id=rule.rule_id,
                )

        # 2. Case-insensitive match against rules
        for rule in rules:
            if rule.source_label.strip().lower() == label_lower:
                node = await self._resolver.get_node(rule.canonical_value)
                return MappingResult(
                    source_label=label_stripped,
                    canonical_value=rule.canonical_value,
                    canonical_name=node.name if node else rule.canonical_value,
                    confidence=rule.confidence * 0.95,
                    match_type=MatchType.CASE_INSENSITIVE,
                    rule_id=rule.rule_id,
                )

        alternatives: list[tuple[str, float]] = []

        # 3. Alias match
        for canonical_value, alias_set in self._aliases.items():
            if label_lower in alias_set:
                node = await self._resolver.get_node(canonical_value)
                if node is not None:
                    alternatives.append((canonical_value, 0.9))

        # 4. Synonym match
        for canonical_value, synonym_set in self._synonyms.items():
            if label_lower in synonym_set:
                node = await self._resolver.get_node(canonical_value)
                if node is not None:
                    alternatives.append((canonical_value, 0.7))

        # 5. Plural normalization
        singular = _singularize(label_lower)
        if singular != label_lower:
            for rule in rules:
                if rule.source_label.strip().lower() == singular:
                    node = await self._resolver.get_node(rule.canonical_value)
                    if node is not None:
                        alternatives.append((rule.canonical_value, 0.85))

        # 6. Regex match
        for pattern, canonical_value, conf in self._regex_rules:
            if pattern.search(label_stripped):
                alternatives.append((canonical_value, conf))

        # 7. Ontology label search (fuzzy name match)
        if not alternatives:
            results = await self._resolver.find_by_label(label_stripped)
            for r in results:
                confidence = _compute_text_similarity(label_lower, r.name.lower())
                if confidence > 0.3:
                    alternatives.append((r.value, confidence))

        if not alternatives:
            return MappingResult(
                source_label=label_stripped,
                canonical_value="unknown_object",
                canonical_name="Unknown Object",
                confidence=0.0,
                match_type=MatchType.SYNONYM,
            )

        alternatives.sort(key=lambda x: x[1], reverse=True)
        best_value, best_conf = alternatives[0]
        best_node = await self._resolver.get_node(best_value)

        return MappingResult(
            source_label=label_stripped,
            canonical_value=best_value,
            canonical_name=best_node.name if best_node else best_value,
            confidence=best_conf,
            match_type=MatchType.SYNONYM if best_conf < 0.9 else MatchType.ALIAS,
            alternatives=tuple(alternatives[1:4]),
        )

    async def map_batch(
        self,
        dataset_name: str,
        labels: list[str],
        rules: list[MappingRule],
    ) -> list[MappingResult]:
        results: list[MappingResult] = []
        for label in labels:
            result = await self.map_label(dataset_name, label, rules)
            results.append(result)
        return results


def _singularize(word: str) -> str:
    """Simple English singularization (not a full NLP pipeline)."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ses"):
        return word[:-2]
    if word.endswith("xes"):
        return word[:-2]
    if word.endswith("shes"):
        return word[:-2]
    if word.endswith("ches"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _compute_text_similarity(a: str, b: str) -> float:
    """Simple character-level similarity for ontology label matching."""
    if not a or not b:
        return 0.0
    set_a = set(a.replace("_", "").replace(" ", ""))
    set_b = set(b.replace("_", "").replace(" ", ""))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
