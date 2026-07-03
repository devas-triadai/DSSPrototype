from __future__ import annotations

import pytest

from backend.dataset_conversion.annotation_converter import AnnotationConverter
from backend.dataset_conversion.models import (
    CoordinateSystem,
    GeometryType,
    SourceAnnotation,
)


class TestAnnotationConverter:
    @pytest.fixture
    def converter(self) -> AnnotationConverter:
        return AnnotationConverter()

    @pytest.mark.asyncio
    async def test_convert_annotation(
        self,
        converter: AnnotationConverter,
        sample_source_annotation: SourceAnnotation,
    ) -> None:
        result = await converter.convert_annotation(
            sample_source_annotation,
            "ground_vehicle.car",
        )
        assert result.canonical_label == "ground_vehicle.car"
        assert result.x == 100.0
        assert result.y == 150.0
        assert result.width == 200.0
        assert result.height == 300.0
        assert result.confidence == 0.95
        assert result.source_label == "car"

    @pytest.mark.asyncio
    async def test_convert_annotation_normalized(
        self,
        converter: AnnotationConverter,
    ) -> None:
        src = SourceAnnotation(
            id="ann002",
            image_id="img002",
            category_id=2,
            category_name="person",
            geometry_type=GeometryType.NORMALIZED,
            coordinates=(0.5, 0.5, 0.3, 0.4),
            coordinate_system=CoordinateSystem.PIXEL,
            image_width=640,
            image_height=480,
        )
        result = await converter.convert_annotation(src, "people.person")
        assert result.canonical_label == "people.person"

    @pytest.mark.asyncio
    async def test_convert_batch(
        self,
        converter: AnnotationConverter,
        sample_source_annotations: list[SourceAnnotation],
    ) -> None:
        label_map = {"car": "ground_vehicle.car", "person": "people.person"}
        results = await converter.convert_batch(
            sample_source_annotations,
            label_map,
        )
        assert len(results) == 10
        assert results[0].canonical_label == "ground_vehicle.car"
        assert results[1].canonical_label == "people.person"

    @pytest.mark.asyncio
    async def test_convert_batch_missing_label(
        self,
        converter: AnnotationConverter,
        sample_source_annotations: list[SourceAnnotation],
    ) -> None:
        label_map = {"car": "ground_vehicle.car"}
        with pytest.raises(Exception):
            await converter.convert_batch(sample_source_annotations, label_map)

    @pytest.mark.asyncio
    async def test_convert_polygon_annotation(
        self,
        converter: AnnotationConverter,
    ) -> None:
        src = SourceAnnotation(
            id="ann003",
            image_id="img003",
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.POLYGON,
            coordinates=(10.0, 20.0, 110.0, 20.0, 110.0, 220.0, 10.0, 220.0),
            coordinate_system=CoordinateSystem.PIXEL,
            image_width=640,
            image_height=480,
        )
        result = await converter.convert_annotation(src, "ground_vehicle.car")
        assert result.geometry_type == GeometryType.POLYGON
        assert result.width == 100.0

    @pytest.mark.asyncio
    async def test_confidence_default(
        self,
        converter: AnnotationConverter,
    ) -> None:
        src = SourceAnnotation(
            id="ann004",
            image_id="img004",
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.BBOX,
            coordinates=(10.0, 20.0, 100.0, 200.0),
            coordinate_system=CoordinateSystem.PIXEL,
        )
        result = await converter.convert_annotation(src, "ground_vehicle.car")
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_source_metadata_preserved(
        self,
        converter: AnnotationConverter,
    ) -> None:
        src = SourceAnnotation(
            id="ann005",
            image_id="img005",
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.BBOX,
            coordinates=(10.0, 20.0, 100.0, 200.0),
            coordinate_system=CoordinateSystem.PIXEL,
            metadata={"iscrowd": "0", "occluded": "false"},
        )
        result = await converter.convert_annotation(src, "ground_vehicle.car")
        assert result.metadata["iscrowd"] == "0"
        assert result.metadata["occluded"] == "false"

    @pytest.mark.asyncio
    async def test_canonical_name_generated(
        self,
        converter: AnnotationConverter,
    ) -> None:
        src = SourceAnnotation(
            id="ann006",
            image_id="img006",
            category_id=1,
            category_name="pickup_truck",
            geometry_type=GeometryType.BBOX,
            coordinates=(10.0, 20.0, 100.0, 200.0),
            coordinate_system=CoordinateSystem.PIXEL,
        )
        result = await converter.convert_annotation(src, "ground_vehicle.pickup_truck")
        assert "Pickup" in result.canonical_name
