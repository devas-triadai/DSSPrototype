from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.dataset_conversion.models import CanonicalAnnotation
from backend.dataset_quality.geometry_validator import GeometryValidator


class TestGeometryValidator:
    @pytest.fixture
    def validator(self) -> GeometryValidator:
        return GeometryValidator()

    @pytest.mark.asyncio
    async def test_valid_annotations(
        self, validator: GeometryValidator, sample_annotations: Sequence[CanonicalAnnotation]
    ) -> None:
        result = await validator.validate(sample_annotations)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_empty_annotations(self, validator: GeometryValidator) -> None:
        result = await validator.validate([])
        assert result.passed is True
        assert result.total_geometries == 0

    @pytest.mark.asyncio
    async def test_invalid_bbox_dimensions(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation.model_construct(
                id="bad_bbox",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=-10.0,
                height=0.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_bbox_count == 1
        assert not result.passed

    @pytest.mark.asyncio
    async def test_invalid_polygon(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="bad_poly",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.POLYGON,
                x=0.0,
                y=0.0,
                width=0.0,
                height=0.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_polygon_count == 1

    @pytest.mark.asyncio
    async def test_invalid_segmentation(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="bad_seg",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.SEGMENTATION,
                x=0.0,
                y=0.0,
                width=0.0,
                height=0.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_segmentation_count == 1

    @pytest.mark.asyncio
    async def test_invalid_obb(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation.model_construct(
                id="bad_obb",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.OBB,
                x=0.0,
                y=0.0,
                width=0.0,
                height=-5.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_rotated_box_count == 1

    @pytest.mark.asyncio
    async def test_negative_pixel_coordinates(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="neg_coord",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=-10.0,
                y=-20.0,
                width=100.0,
                height=100.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_pixel_coord_count == 1

    @pytest.mark.asyncio
    async def test_mixed_geometry_types(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="a",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            ),
            CanonicalAnnotation(
                id="b",
                image_id="img001",
                canonical_label="person",
                canonical_name="person",
                geometry_type=GeometryType.POLYGON,
                x=0.0,
                y=0.0,
                width=50.0,
                height=50.0,
            ),
            CanonicalAnnotation(
                id="c",
                image_id="img001",
                canonical_label="drone",
                canonical_name="drone",
                geometry_type=GeometryType.OBB,
                x=0.0,
                y=0.0,
                width=20.0,
                height=30.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_bbox_count == 0
        assert result.invalid_polygon_count == 0
        assert result.invalid_rotated_box_count == 0

    @pytest.mark.asyncio
    async def test_valid_bbox_is_ok(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="valid",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=10.0,
                y=20.0,
                width=100.0,
                height=200.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_bbox_count == 0

    @pytest.mark.asyncio
    async def test_valid_obb_is_ok(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="valid_obb",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.OBB,
                x=0.0,
                y=0.0,
                width=50.0,
                height=100.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_rotated_box_count == 0

    @pytest.mark.asyncio
    async def test_valid_polygon_is_ok(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="valid_poly",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.POLYGON,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_polygon_count == 0

    @pytest.mark.asyncio
    async def test_valid_segmentation_is_ok(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="valid_seg",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.SEGMENTATION,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_segmentation_count == 0

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid(self, validator: GeometryValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = [
            CanonicalAnnotation(
                id="g1",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            ),
            CanonicalAnnotation.model_construct(
                id="b1",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=0.0,
                height=0.0,
            ),
            CanonicalAnnotation.model_construct(
                id="b2",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=-10.0,
                height=20.0,
            ),
        ]
        result = await validator.validate(anns)
        assert result.invalid_bbox_count >= 2

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import GeometryValidatorInterface

        assert issubclass(GeometryValidator, GeometryValidatorInterface)
