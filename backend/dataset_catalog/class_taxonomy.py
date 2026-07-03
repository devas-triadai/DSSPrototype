"""Taxonomy management for the military domain classification system.

Provides a tree-based taxonomy structure (loaded from JSON or ontology service)
and methods to analyze coverage of dataset classes against the taxonomy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import TaxonomyNodeNotFoundError
from backend.dataset_catalog.interfaces import TaxonomicalCoverageInterface
from backend.dataset_catalog.models import (
    CoverageReport,
    DatasetProfile,
    TaxonomicalCoverage,
    TaxonomyNode,
)

logger = logging.getLogger("dss.dataset_catalog.taxonomy")


class ClassTaxonomy(TaxonomicalCoverageInterface):
    """Manages a hierarchical taxonomy of military-domain classes.

    Can load from:
      1. A JSON file on disk (backup/fast path)
      2. The ontology service (preferred, when available)

    The taxonomy is organized as a tree where each node can have children,
    enabling multi-level coverage analysis.
    """

    def __init__(
        self,
        taxonomy_path: Path | None = None,
        ontology_service: object | None = None,
    ) -> None:
        self._taxonomy_path = taxonomy_path or dc_config.taxonomy_path
        self._ontology_service = ontology_service
        self._nodes: dict[str, TaxonomyNode] = {}
        self._root_ids: list[str] = []
        self._version: str = ""
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._taxonomy_path and self._taxonomy_path.exists():
            try:
                with self._taxonomy_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._version = data.get("version", "1.0.0")
                raw_nodes = data.get("nodes", [])
                for node_data in raw_nodes:
                    node = TaxonomyNode(**node_data)
                    self._nodes[node.node_id] = node
                self._root_ids = [
                    n.node_id
                    for n in self._nodes.values()
                    if n.parent_id is None
                ]
                logger.info(
                    "Loaded taxonomy | nodes=%d | roots=%d | version=%s",
                    len(self._nodes),
                    len(self._root_ids),
                    self._version,
                )
                return
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load taxonomy: %s", exc)

        self._build_default_taxonomy()

    def _build_default_taxonomy(self) -> None:
        """Build a default military taxonomy tree programmatically."""
        nodes: list[TaxonomyNode] = [
            TaxonomyNode(
                node_id="ground_vehicles",
                name="Ground Vehicles",
                parent_id=None,
                domain="military",
                description="All ground-based military vehicles",
                keywords=["tank", "armored", "vehicle", "truck", "jeep"],
                aliases=["ground_vehicles", "land_vehicles"],
                priority=1.0,
                child_ids=["tanks", "armored_personnel_carriers", "utility_vehicles"],
            ),
            TaxonomyNode(
                node_id="tanks",
                name="Tanks",
                parent_id="ground_vehicles",
                domain="military",
                description="Main battle tanks and light tanks",
                keywords=["tank", "main battle tank", "mbt", "armor"],
                aliases=["main_battle_tank", "battle_tank"],
                priority=1.0,
                child_ids=["mbt", "light_tank"],
            ),
            TaxonomyNode(
                node_id="mbt",
                name="Main Battle Tank",
                parent_id="tanks",
                domain="military",
                description="Main battle tanks like M1 Abrams, T-90, Leopard 2",
                keywords=["m1 abrams", "t-90", "leopard 2", "challenger 2"],
                aliases=["main_battle_tank", "heavy_tank"],
                priority=1.0,
            ),
            TaxonomyNode(
                node_id="light_tank",
                name="Light Tank",
                parent_id="tanks",
                domain="military",
                description="Light reconnaissance tanks",
                keywords=["scout", "light tank", "recon"],
                aliases=["scout_tank"],
                priority=0.6,
            ),
            TaxonomyNode(
                node_id="armored_personnel_carriers",
                name="Armored Personnel Carriers",
                parent_id="ground_vehicles",
                domain="military",
                description="Vehicles designed to transport infantry",
                keywords=["apc", "armored", "personnel carrier"],
                aliases=["apc", "armored_personnel_carrier"],
                priority=0.9,
                child_ids=["ifv", "mrap"],
            ),
            TaxonomyNode(
                node_id="ifv",
                name="Infantry Fighting Vehicle",
                parent_id="armored_personnel_carriers",
                domain="military",
                description="Infantry fighting vehicles like BMP, Bradley",
                keywords=["ifv", "bmp", "bradley", "infantry fighting vehicle"],
                aliases=["infantry_fighting_vehicle"],
                priority=0.9,
            ),
            TaxonomyNode(
                node_id="mrap",
                name="MRAP",
                parent_id="armored_personnel_carriers",
                domain="military",
                description="Mine-Resistant Ambush Protected vehicles",
                keywords=["mrap", "mine resistant", "ambush protected"],
                aliases=["mine_resistant"],
                priority=0.7,
            ),
            TaxonomyNode(
                node_id="utility_vehicles",
                name="Utility Vehicles",
                parent_id="ground_vehicles",
                domain="military",
                description="Support and utility military vehicles",
                keywords=["truck", "humvee", "jeep", "utility", "supply"],
                aliases=["support_vehicle", "utility_truck"],
                priority=0.7,
            ),
            TaxonomyNode(
                node_id="aircraft",
                name="Aircraft",
                parent_id=None,
                domain="military",
                description="All military aircraft",
                keywords=["aircraft", "plane", "jet", "helicopter", "drone", "uav"],
                aliases=["military_aircraft", "warplane"],
                priority=1.0,
                child_ids=["fighter_jet", "helicopter", "uav", "bomber"],
            ),
            TaxonomyNode(
                node_id="fighter_jet",
                name="Fighter Jet",
                parent_id="aircraft",
                domain="military",
                description="Fighter and multirole aircraft",
                keywords=["fighter", "jet", "f-16", "f-35", "su-27", "mig"],
                aliases=["fighter_aircraft", "multirole_fighter"],
                priority=1.0,
            ),
            TaxonomyNode(
                node_id="helicopter",
                name="Helicopter",
                parent_id="aircraft",
                domain="military",
                description="Attack and transport helicopters",
                keywords=["helicopter", "chopper", "apache", "black hawk"],
                aliases=["attack_helicopter", "transport_helicopter"],
                priority=0.8,
            ),
            TaxonomyNode(
                node_id="uav",
                name="Unmanned Aerial Vehicle",
                parent_id="aircraft",
                domain="military",
                description="Drones and unmanned aerial systems",
                keywords=["uav", "drone", "unmanned", "quadcopter"],
                aliases=["drone", "uas", "unmanned_aerial_vehicle"],
                priority=0.9,
            ),
            TaxonomyNode(
                node_id="bomber",
                name="Bomber",
                parent_id="aircraft",
                domain="military",
                description="Strategic and tactical bomber aircraft",
                keywords=["bomber", "b-52", "strategic bomber"],
                aliases=["strategic_bomber"],
                priority=0.6,
            ),
            TaxonomyNode(
                node_id="naval_vessels",
                name="Naval Vessels",
                parent_id=None,
                domain="military",
                description="All military naval vessels",
                keywords=["ship", "vessel", "navy", "warship", "carrier"],
                aliases=["warship", "naval_ship"],
                priority=0.8,
                child_ids=["aircraft_carrier", "destroyer", "submarine"],
            ),
            TaxonomyNode(
                node_id="aircraft_carrier",
                name="Aircraft Carrier",
                parent_id="naval_vessels",
                domain="military",
                description="Aircraft carriers and amphibious assault ships",
                keywords=["carrier", "aircraft carrier", "naval aviation"],
                aliases=["carrier"],
                priority=0.8,
            ),
            TaxonomyNode(
                node_id="destroyer",
                name="Destroyer",
                parent_id="naval_vessels",
                domain="military",
                description="Destroyers, frigates, and corvettes",
                keywords=["destroyer", "frigate", "corvette", "warship"],
                aliases=["frigate", "corvette"],
                priority=0.7,
            ),
            TaxonomyNode(
                node_id="submarine",
                name="Submarine",
                parent_id="naval_vessels",
                domain="military",
                description="Submarines and underwater vessels",
                keywords=["submarine", "sub", "underwater"],
                aliases=["sub", "u-boat"],
                priority=0.7,
            ),
            TaxonomyNode(
                node_id="personnel",
                name="Military Personnel",
                parent_id=None,
                domain="military",
                description="Individual soldiers and personnel",
                keywords=["soldier", "personnel", "troop", "infantry"],
                aliases=["soldier", "troops"],
                priority=0.7,
            ),
        ]
        for node in nodes:
            self._nodes[node.node_id] = node
        self._root_ids = [n.node_id for n in nodes if n.parent_id is None]
        self._version = "1.0.0"
        logger.info(
            "Built default taxonomy | nodes=%d | roots=%d | version=%s",
            len(self._nodes),
            len(self._root_ids),
            self._version,
        )

    # ------------------------------------------------------------------
    # Taxonomy querying
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> TaxonomyNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise TaxonomyNodeNotFoundError(f"Taxonomy node not found: {node_id}")
        return node

    def get_children(self, node_id: str) -> list[TaxonomyNode]:
        node = self.get_node(node_id)
        return [self._nodes[cid] for cid in node.child_ids if cid in self._nodes]

    def get_descendants(self, node_id: str) -> list[TaxonomyNode]:
        result: list[TaxonomyNode] = []
        node = self.get_node(node_id)
        for cid in node.child_ids:
            child = self._nodes.get(cid)
            if child:
                result.append(child)
                result.extend(self.get_descendants(cid))
        return result

    def get_ancestors(self, node_id: str) -> list[TaxonomyNode]:
        result: list[TaxonomyNode] = []
        node = self.get_node(node_id)
        while node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent:
                result.append(parent)
                node = parent
            else:
                break
        return result

    def find_nodes_by_keyword(self, keyword: str) -> list[TaxonomyNode]:
        kw = keyword.lower().replace("_", " ").replace("-", " ")
        matches: list[TaxonomyNode] = []
        for node in self._nodes.values():
            if kw in node.name.lower():
                matches.append(node)
                continue
            if any(kw in k.lower() for k in node.keywords):
                matches.append(node)
                continue
            if any(kw in a.lower().replace("_", " ").replace("-", " ") for a in node.aliases):
                matches.append(node)
                continue
        return matches

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    def load_taxonomy(self, path: Path | None = None) -> list[TaxonomyNode]:
        if path:
            self._taxonomy_path = path
        self._nodes.clear()
        self._load()
        return list(self._nodes.values())

    def analyze_coverage(
        self,
        profile_classes: Sequence[str],
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> TaxonomicalCoverage:
        nodes = taxonomy or list(self._nodes.values())
        if not nodes:
            nodes = list(self._nodes.values())

        profile_list = list(profile_classes)
        profile_keywords = set()
        for cls in profile_list:
            parts = cls.lower().replace("_", " ").replace("-", " ").split()
            profile_keywords.update(parts)
            profile_keywords.add(cls.lower().replace(" ", "_"))

        covered: list[str] = []
        uncovered: list[str] = []
        partial: list[str] = []
        domain_counts: dict[str, dict[str, int]] = {}

        for node in nodes:
            if node.domain not in domain_counts:
                domain_counts[node.domain] = {"covered": 0, "partial": 0, "total": 0}
            domain_counts[node.domain]["total"] += 1

            node_keywords_lower = {k.lower() for k in node.keywords}
            node_aliases_lower = {
                a.lower().replace("_", " ").replace("-", " ")
                for a in node.aliases
            }
            node_name_lower = node.name.lower()

            # Check exact alias match
            exact_alias_match = any(
                a in profile_keywords for a in node_aliases_lower
            )
            # Check keyword match
            keyword_match = bool(node_keywords_lower & profile_keywords)
            # Check name match
            name_match = node_name_lower in profile_keywords or any(
                node_name_lower == pk for pk in profile_keywords
            )

            if name_match or exact_alias_match or (
                keyword_match and any(
                    kw in profile_keywords for kw in node_keywords_lower
                )
            ):
                covered.append(node.node_id)
                domain_counts[node.domain]["covered"] += 1
            elif keyword_match:
                partial.append(node.node_id)
                domain_counts[node.domain]["partial"] += 1
            else:
                uncovered.append(node.node_id)

        total = len(nodes)
        coverage_ratio = len(covered) / total if total > 0 else 0.0

        domain_coverage: dict[str, float] = {}
        for domain, counts in domain_counts.items():
            domain_coverage[domain] = (
                counts["covered"] / counts["total"] if counts["total"] > 0 else 0.0
            )

        return TaxonomicalCoverage(
            taxonomy_version=self._version,
            total_nodes=total,
            covered_nodes=covered,
            uncovered_nodes=uncovered,
            partial_nodes=partial,
            coverage_ratio=coverage_ratio,
            domain_coverage=domain_coverage,
        )

    def get_coverage_report(
        self,
        profile: DatasetProfile,
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> CoverageReport:
        coverage = self.analyze_coverage(
            list(profile.classes), taxonomy
        )
        domain_breakdown = coverage.domain_coverage
        sorted_domains = sorted(
            domain_breakdown.items(), key=lambda x: x[1]
        )
        weakest = [d for d, _ in sorted_domains[:3]]
        strongest = [d for d, _ in sorted_domains[-3:]]

        return CoverageReport(
            report_id=f"cr_{profile.profile_id}",
            total_entries_analyzed=1,
            entries=[coverage],
            aggregate_coverage=coverage.coverage_ratio,
            domain_breakdown=domain_breakdown,
            weakest_domains=weakest,
            strongest_domains=strongest,
        )
