"""Tests for OntologyResolver."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.exceptions import OntologyNodeNotFoundError
from backend.ontology_mapping.ontology_resolver import OntologyResolver


@pytest.fixture
def resolver() -> OntologyResolver:
    OntologyResolver._tree = None
    return OntologyResolver()


@pytest.mark.asyncio
async def test_resolve_root(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("root")
    assert res.resolved_node is not None
    assert res.resolved_node.value == "root"
    assert res.resolved_node.depth == 0


@pytest.mark.asyncio
async def test_resolve_category(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("ground_vehicle")
    assert res.resolved_node is not None
    assert res.resolved_node.name == "Ground Vehicle"
    assert res.resolved_node.depth == 1
    assert res.resolved_node.parent == "root"


@pytest.mark.asyncio
async def test_resolve_class(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("ground_vehicle.car")
    assert res.resolved_node is not None
    assert res.resolved_node.name == "Car"
    assert res.resolved_node.depth == 2
    assert res.resolved_node.parent == "ground_vehicle"


@pytest.mark.asyncio
async def test_resolve_nonexistent(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("nonexistent")
    assert res.resolved_node is None


@pytest.mark.asyncio
async def test_resolve_people(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("people.person")
    assert res.resolved_node is not None
    assert res.resolved_node.name == "Person"


@pytest.mark.asyncio
async def test_get_node_class(resolver: OntologyResolver) -> None:
    node = await resolver.get_node("ground_vehicle.car")
    assert node is not None
    assert node.category == "ground_vehicle"
    assert node.is_leaf


@pytest.mark.asyncio
async def test_get_node_nonexistent(resolver: OntologyResolver) -> None:
    node = await resolver.get_node("nonexistent")
    assert node is None


@pytest.mark.asyncio
async def test_get_children(resolver: OntologyResolver) -> None:
    children = await resolver.get_children("ground_vehicle")
    assert len(children) > 0
    child_values = {c.value for c in children}
    assert "ground_vehicle.car" in child_values


@pytest.mark.asyncio
async def test_get_children_nonexistent(resolver: OntologyResolver) -> None:
    with pytest.raises(OntologyNodeNotFoundError):
        await resolver.get_children("nonexistent")


@pytest.mark.asyncio
async def test_get_parent(resolver: OntologyResolver) -> None:
    parent = await resolver.get_parent("ground_vehicle.car")
    assert parent is not None
    assert parent.value == "ground_vehicle"


@pytest.mark.asyncio
async def test_get_parent_root(resolver: OntologyResolver) -> None:
    parent = await resolver.get_parent("root")
    assert parent is None


@pytest.mark.asyncio
async def test_get_ancestors(resolver: OntologyResolver) -> None:
    ancestors = await resolver.get_ancestors("ground_vehicle.car")
    values = [a.value for a in ancestors]
    assert "ground_vehicle" in values
    assert "root" in values


@pytest.mark.asyncio
async def test_get_full_ancestry(resolver: OntologyResolver) -> None:
    ancestors = await resolver.get_ancestors("ground_vehicle.car")
    assert len(ancestors) >= 2
    assert ancestors[0].value == "root"


@pytest.mark.asyncio
async def test_get_ancestors_nonexistent(
    resolver: OntologyResolver,
) -> None:
    with pytest.raises(OntologyNodeNotFoundError):
        await resolver.get_ancestors("nonexistent")


@pytest.mark.asyncio
async def test_get_siblings(resolver: OntologyResolver) -> None:
    siblings = await resolver.get_siblings("ground_vehicle.car")
    sibling_values = {s.value for s in siblings}
    assert "ground_vehicle.truck" in sibling_values


@pytest.mark.asyncio
async def test_get_siblings_root(resolver: OntologyResolver) -> None:
    siblings = await resolver.get_siblings("root")
    assert siblings == []


@pytest.mark.asyncio
async def test_get_siblings_category(resolver: OntologyResolver) -> None:
    siblings = await resolver.get_siblings("ground_vehicle")
    assert len(siblings) > 0
    cat_names = {s.name for s in siblings}
    assert "People" in cat_names


@pytest.mark.asyncio
async def test_get_leaves(resolver: OntologyResolver) -> None:
    leaves = await resolver.get_leaves()
    assert len(leaves) > 0
    for leaf in leaves:
        assert leaf.is_leaf


@pytest.mark.asyncio
async def test_contains(resolver: OntologyResolver) -> None:
    assert await resolver.contains("ground_vehicle.car")
    assert not await resolver.contains("nonexistent")


@pytest.mark.asyncio
async def test_all_categories_present(resolver: OntologyResolver) -> None:
    categories = {"people", "ground_vehicle", "aircraft", "watercraft",
                  "buildings", "infrastructure", "road_network", "vegetation",
                  "water_bodies", "terrain", "smoke", "fire",
                  "construction", "engineering", "utilities", "barriers",
                  "bridges", "airfields", "ports", "rail",
                  "natural", "ooi"}
    for cat in categories:
        node = await resolver.get_node(cat)
        assert node is not None, f"Category '{cat}' not found"
        assert node.depth == 1


@pytest.mark.asyncio
async def test_find_by_label(resolver: OntologyResolver) -> None:
    results = await resolver.find_by_label("car")
    assert len(results) >= 1
    assert any(r.value == "ground_vehicle.car" for r in results)


@pytest.mark.asyncio
async def test_find_by_label_no_results(
    resolver: OntologyResolver,
) -> None:
    results = await resolver.find_by_label("zzzznotfound")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_resolve_containing_siblings(
    resolver: OntologyResolver,
) -> None:
    res = await resolver.resolve("people.person")
    assert res.siblings is not None
    sibling_values = {s.value for s in res.siblings}
    assert "people.group" in sibling_values
    assert "people.crowd" in sibling_values


@pytest.mark.asyncio
async def test_resolve_path(resolver: OntologyResolver) -> None:
    res = await resolver.resolve("ground_vehicle.car")
    assert len(res.path) >= 2
    assert res.path[0] == "root"


@pytest.mark.asyncio
async def test_all_ontology_values_have_depth(
    resolver: OntologyResolver,
) -> None:
    for value in ["ground_vehicle.car", "people.person",
                  "aircraft.uav", "buildings.building",
                  "watercraft.ship", "infrastructure.pipeline"]:
        node = await resolver.get_node(value)
        assert node is not None, f"Node '{value}' not found"
        assert node.depth == 2


@pytest.mark.asyncio
async def test_unknown_object_leaf(resolver: OntologyResolver) -> None:
    node = await resolver.get_node("unknown_object")
    assert node is not None
    assert node.is_leaf
