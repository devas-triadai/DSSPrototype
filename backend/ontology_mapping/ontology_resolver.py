"""Ontology tree builder and resolver.

Builds the ontology tree programmatically from the frozen ObjectType enum.
The tree is constructed once and cached for the lifetime of the process.
"""

from __future__ import annotations

from typing import Final

from backend.contracts.enums.core import ObjectType
from backend.ontology_mapping.exceptions import OntologyNodeNotFoundError
from backend.ontology_mapping.models import OntologyNode, OntologyResolution

_CATEGORY_NAMES: Final[dict[str, str]] = {
    "unknown_object": "Unknown Object",
    "people": "People",
    "ground_vehicle": "Ground Vehicle",
    "aircraft": "Aircraft",
    "watercraft": "Watercraft",
    "buildings": "Buildings",
    "infrastructure": "Infrastructure",
    "road_network": "Road Network",
    "vegetation": "Vegetation",
    "water_bodies": "Water Bodies",
    "terrain": "Terrain",
    "smoke": "Smoke",
    "fire": "Fire",
    "construction": "Construction",
    "engineering": "Engineering Structures",
    "utilities": "Utilities",
    "barriers": "Barriers",
    "bridges": "Bridges",
    "airfields": "Airfields",
    "ports": "Ports",
    "rail": "Rail Infrastructure",
    "natural": "Natural Features",
    "ooi": "Objects of Interest",
}


def _split_snake(name: str) -> str:
    """Convert 'heavy_equipment' to 'Heavy Equipment'."""
    return name.replace("_", " ").title()


def _extract_class_name(value: str) -> str:
    """Extract human-readable class name from a dotted ontology value."""
    parts = value.split(".")
    if len(parts) == 1:
        return _CATEGORY_NAMES.get(parts[0], _split_snake(parts[0]))
    return _split_snake(parts[-1])


def _extract_category(value: str) -> str:
    """Extract the category key from a dotted ontology value."""
    parts = value.split(".")
    return parts[0] if len(parts) > 1 else value


def _build_tree() -> dict[str, OntologyNode]:
    """Build the full ontology tree from the frozen ObjectType enum."""
    nodes: dict[str, OntologyNode] = {}

    root_node = OntologyNode(
        value="root",
        name="Root",
        category="root",
        category_name="Root",
        parent=None,
        children=frozenset(),
        depth=0,
        is_leaf=False,
    )
    nodes["root"] = root_node

    category_children: dict[str, set[str]] = {}
    class_nodes: dict[str, OntologyNode] = {}

    for member in ObjectType:
        dotted = member.value
        parts = dotted.split(".")
        category_key = parts[0]

        if category_key not in category_children:
            category_children[category_key] = set()

        if len(parts) >= 2:
            class_key = parts[1]
            category_children[category_key].add(class_key)

        parent_value = category_key
        if len(parts) > 2:
            for i in range(2, len(parts)):
                parent_part = ".".join(parts[:i])
                if parent_part not in class_nodes:
                    class_nodes[parent_part] = OntologyNode(
                        value=parent_part,
                        name=_split_snake(parts[i - 1]),
                        category=category_key,
                        category_name=_CATEGORY_NAMES.get(
                            category_key, _split_snake(category_key)
                        ),
                        parent=".".join(parts[: i - 1]) if i > 1 else category_key,
                        children=frozenset(),
                        depth=2 + (i - 2),
                        is_leaf=True,
                        enum_member_name=member.name,
                    )

        if dotted not in class_nodes:
            class_name = _extract_class_name(dotted)
            is_unknown = parts[-1] == "unknown"
            if is_unknown and len(parts) == 2:
                cat_label = _CATEGORY_NAMES.get(category_key, _split_snake(category_key))
                class_name = f"Unknown {cat_label}"

            class_nodes[dotted] = OntologyNode(
                value=dotted,
                name=class_name,
                category=category_key,
                category_name=_CATEGORY_NAMES.get(
                    category_key, _split_snake(category_key)
                ),
                parent=parent_value,
                children=frozenset(),
                depth=2 if len(parts) == 2 else 3,
                is_leaf=True,
                enum_member_name=member.name,
            )

    for category_key, child_names in category_children.items():
        category_name = _CATEGORY_NAMES.get(
            category_key, _split_snake(category_key)
        )
        cat_children = frozenset(
            f"{category_key}.{c}" for c in child_names
        )
        category_node = OntologyNode(
            value=category_key,
            name=category_name,
            category=category_key,
            category_name=category_name,
            parent="root",
            children=cat_children,
            depth=1,
            is_leaf=False,
        )
        nodes[category_key] = category_node

        for child_name in child_names:
            child_value = f"{category_key}.{child_name}"
            if child_value in class_nodes:
                node = class_nodes[child_value]
                node = OntologyNode(
                    value=node.value,
                    name=node.name,
                    category=node.category,
                    category_name=node.category_name,
                    parent=category_key,
                    children=node.children,
                    depth=2,
                    is_leaf=len(node.children) == 0,
                    enum_member_name=node.enum_member_name,
                )
                nodes[child_value] = node

    for value, node in list(nodes.items()):
        if value == "root":
            continue
        if len(node.children) == 0 and node.is_leaf is False:
            nodes[value] = OntologyNode(
                value=node.value,
                name=node.name,
                category=node.category,
                category_name=node.category_name,
                parent=node.parent,
                children=node.children,
                depth=node.depth,
                is_leaf=True,
                enum_member_name=node.enum_member_name,
            )

    category_set = frozenset(
        v for v in nodes.keys()
        if v != "root" and nodes[v].depth == 1
    )
    nodes["root"] = OntologyNode(
        value="root",
        name="Root",
        category="root",
        category_name="Root",
        parent=None,
        children=category_set,
        depth=0,
        is_leaf=False,
    )

    return nodes


class OntologyResolver:
    """Navigates the frozen ontology tree.

    The tree is built once from ``ObjectType`` and cached.
    All traversal operations are O(1) or O(children).
    """

    _tree: dict[str, OntologyNode] | None = None

    @classmethod
    def _ensure_tree(cls) -> dict[str, OntologyNode]:
        if cls._tree is None:
            cls._tree = _build_tree()
        return cls._tree

    @classmethod
    def rebuild(cls) -> None:
        cls._tree = _build_tree()

    async def resolve(self, value: str) -> OntologyResolution:
        tree = self._ensure_tree()
        node = tree.get(value)
        if node is None:
            return OntologyResolution(
                query=value,
                resolved_node=None,
            )

        parent = tree.get(node.parent) if node.parent else None
        children: list[OntologyNode] = [
            tree[c] for c in node.children if c in tree
        ]
        ancestors: list[OntologyNode] = []
        current = node
        while current.parent and current.parent in tree:
            p = tree[current.parent]
            ancestors.append(p)
            current = p
        ancestors.reverse()

        siblings: list[OntologyNode] = []
        if node.parent and node.parent in tree:
            parent_node = tree[node.parent]
            siblings = [
                tree[c]
                for c in parent_node.children
                if c != value and c in tree
            ]

        path_parts: list[str] = []
        current = node
        path_parts.append(current.value)
        while current.parent and current.parent in tree:
            current = tree[current.parent]
            path_parts.append(current.value)
        path_parts.reverse()

        return OntologyResolution(
            query=value,
            resolved_node=node,
            parent=parent,
            children=tuple(children),
            ancestors=tuple(ancestors),
            siblings=tuple(siblings),
            path=tuple(path_parts),
        )

    async def get_node(self, value: str) -> OntologyNode | None:
        tree = self._ensure_tree()
        return tree.get(value)

    async def get_children(self, value: str) -> list[OntologyNode]:
        tree = self._ensure_tree()
        node = tree.get(value)
        if node is None:
            raise OntologyNodeNotFoundError(f"Node '{value}' not found")
        return [tree[c] for c in node.children if c in tree]

    async def get_parent(self, value: str) -> OntologyNode | None:
        tree = self._ensure_tree()
        node = tree.get(value)
        if node is None or node.parent is None:
            return None
        return tree.get(node.parent)

    async def get_ancestors(self, value: str) -> list[OntologyNode]:
        tree = self._ensure_tree()
        node = tree.get(value)
        if node is None:
            raise OntologyNodeNotFoundError(f"Node '{value}' not found")
        result: list[OntologyNode] = []
        current = node
        while current.parent and current.parent in tree:
            p = tree[current.parent]
            result.append(p)
            current = p
        result.reverse()
        return result

    async def get_siblings(self, value: str) -> list[OntologyNode]:
        tree = self._ensure_tree()
        node = tree.get(value)
        if node is None:
            raise OntologyNodeNotFoundError(f"Node '{value}' not found")
        if node.parent is None or node.parent not in tree:
            return []
        parent_node = tree[node.parent]
        return [
            tree[c] for c in parent_node.children if c != value and c in tree
        ]

    async def get_leaves(self) -> list[OntologyNode]:
        tree = self._ensure_tree()
        return [n for n in tree.values() if n.is_leaf]

    async def contains(self, value: str) -> bool:
        tree = self._ensure_tree()
        return value in tree

    async def find_by_label(self, label: str) -> list[OntologyNode]:
        """Search ontology nodes by partial name match (case-insensitive)."""
        tree = self._ensure_tree()
        label_lower = label.lower()
        results: list[OntologyNode] = []
        for node in tree.values():
            if node.value == "root":
                continue
            if label_lower in node.name.lower():
                results.append(node)
            elif label_lower in node.value.lower():
                results.append(node)
        return results
