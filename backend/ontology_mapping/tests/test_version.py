"""Tests for MappingVersionManager."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.mapping_version import MappingVersionManager
from backend.ontology_mapping.models import MappingVersion


@pytest.fixture
def vmanager() -> MappingVersionManager:
    return MappingVersionManager("1.0.0")


@pytest.mark.asyncio
async def test_initial_version(vmanager: MappingVersionManager) -> None:
    assert vmanager.current_version == "1.0.0"


@pytest.mark.asyncio
async def test_bump_major(vmanager: MappingVersionManager) -> None:
    v = await vmanager.bump_major()
    assert v == "2.0.0"
    assert vmanager.current_version == "2.0.0"


@pytest.mark.asyncio
async def test_bump_minor(vmanager: MappingVersionManager) -> None:
    v = await vmanager.bump_minor()
    assert v == "1.1.0"
    assert vmanager.current_version == "1.1.0"


@pytest.mark.asyncio
async def test_bump_patch(vmanager: MappingVersionManager) -> None:
    v = await vmanager.bump_patch()
    assert v == "1.0.1"
    assert vmanager.current_version == "1.0.1"


@pytest.mark.asyncio
async def test_bump_chain(vmanager: MappingVersionManager) -> None:
    await vmanager.bump_minor()
    await vmanager.bump_patch()
    assert vmanager.current_version == "1.1.1"


@pytest.mark.asyncio
async def test_compare_equal() -> None:
    vm = MappingVersionManager()
    assert await vm.compare("1.0.0", "1.0.0") == 0


@pytest.mark.asyncio
async def test_compare_less() -> None:
    vm = MappingVersionManager()
    assert await vm.compare("1.0.0", "2.0.0") == -1


@pytest.mark.asyncio
async def test_compare_greater() -> None:
    vm = MappingVersionManager()
    assert await vm.compare("2.0.0", "1.0.0") == 1


@pytest.mark.asyncio
async def test_compare_patch() -> None:
    vm = MappingVersionManager()
    assert await vm.compare("1.0.1", "1.0.2") == -1
    assert await vm.compare("1.0.5", "1.0.4") == 1


@pytest.mark.asyncio
async def test_compare_minor() -> None:
    vm = MappingVersionManager()
    assert await vm.compare("1.1.0", "1.2.0") == -1
    assert await vm.compare("1.3.0", "1.2.0") == 1


@pytest.mark.asyncio
async def test_is_compatible_same_major() -> None:
    vm = MappingVersionManager()
    mv = MappingVersion(
        version="1.0.0",
        ontology_version="1.5.0",
    )
    assert await vm.is_compatible(mv, "1.2.0")


@pytest.mark.asyncio
async def test_is_compatible_diff_major() -> None:
    vm = MappingVersionManager()
    mv = MappingVersion(
        version="1.0.0",
        ontology_version="1.0.0",
    )
    assert not await vm.is_compatible(mv, "2.0.0")


@pytest.mark.asyncio
async def test_parse_version() -> None:
    assert MappingVersionManager._parse("1.2.3") == (1, 2, 3)
    assert MappingVersionManager._parse("0.0.1") == (0, 0, 1)
    assert MappingVersionManager._parse("10.20.30") == (10, 20, 30)


@pytest.mark.asyncio
async def test_parse_incomplete() -> None:
    assert MappingVersionManager._parse("1") == (1, 0, 0)
    assert MappingVersionManager._parse("1.2") == (1, 2, 0)
    assert MappingVersionManager._parse("0") == (0, 0, 0)


@pytest.mark.asyncio
async def test_custom_initial_version() -> None:
    vm = MappingVersionManager("2.3.4")
    assert vm.current_version == "2.3.4"


@pytest.mark.asyncio
async def test_bump_from_custom() -> None:
    vm = MappingVersionManager("2.3.4")
    await vm.bump_minor()
    assert vm.current_version == "2.4.0"


@pytest.mark.asyncio
async def test_multiple_bumps() -> None:
    vm = MappingVersionManager("0.9.0")
    await vm.bump_minor()
    await vm.bump_patch()
    await vm.bump_patch()
    assert vm.current_version == "0.10.2"
