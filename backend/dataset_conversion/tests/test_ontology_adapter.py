from __future__ import annotations

import pytest

from backend.dataset_conversion.ontology_adapter import OntologyAdapter
from backend.ontology_mapping import OntologyMappingService


class TestOntologyAdapter:
    @pytest.fixture
    def adapter(self) -> OntologyAdapter:
        return OntologyAdapter()

    @pytest.mark.asyncio
    async def test_translate_unknown_label_falls_back(
        self,
        adapter: OntologyAdapter,
    ) -> None:
        result = await adapter.translate_label("nonexistent_label", "test_ds")
        assert result == "unknown_object"

    @pytest.mark.asyncio
    async def test_is_known_label_true(self, adapter: OntologyAdapter) -> None:
        known = await adapter.is_known_label("people.person")
        assert known is True

    @pytest.mark.asyncio
    async def test_is_known_label_false(self, adapter: OntologyAdapter) -> None:
        known = await adapter.is_known_label("nonexistent.node")
        assert known is False

    @pytest.mark.asyncio
    async def test_mapping_service_property(
        self,
        adapter: OntologyAdapter,
    ) -> None:
        svc = adapter.mapping_service
        assert isinstance(svc, OntologyMappingService)

    @pytest.mark.asyncio
    async def test_translate_batch(
        self,
        adapter: OntologyAdapter,
    ) -> None:
        result = await adapter.translate_batch(
            ["car", "person", "unknown_xyz"],
            "test_batch",
        )
        assert "car" in result
        assert "person" in result
        assert "unknown_xyz" in result

    @pytest.mark.asyncio
    async def test_translate_batch_unknown_falls_back(
        self,
        adapter: OntologyAdapter,
    ) -> None:
        result = await adapter.translate_batch(
            ["zxy_nonexistent"],
            "test_batch",
        )
        assert result["zxy_nonexistent"] == "unknown_object"
