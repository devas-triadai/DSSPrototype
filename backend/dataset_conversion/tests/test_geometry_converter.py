from __future__ import annotations

import pytest

from backend.dataset_conversion.exceptions import GeometryError
from backend.dataset_conversion.geometry_converter import GeometryConverter


class TestGeometryConverter:
    @pytest.fixture
    def converter(self) -> GeometryConverter:
        return GeometryConverter()

    @pytest.mark.asyncio
    async def test_xyxy_to_xywh(self, converter: GeometryConverter) -> None:
        result = await converter.to_canonical_bbox(
            (10, 20, 110, 220),
            "xyxy",
        )
        assert result == (10, 20, 100, 200)

    @pytest.mark.asyncio
    async def test_xywh_to_xywh(self, converter: GeometryConverter) -> None:
        result = await converter.to_canonical_bbox(
            (10, 20, 100, 200),
            "xywh",
        )
        assert result == (10, 20, 100, 200)

    @pytest.mark.asyncio
    async def test_cxcywh_with_image_dims(
        self,
        converter: GeometryConverter,
    ) -> None:
        result = await converter.to_canonical_bbox(
            (0.5, 0.5, 0.3, 0.4),
            "cxcywh",
            image_width=640,
            image_height=480,
        )
        assert result == (224.0, 144.0, 192.0, 192.0)

    @pytest.mark.asyncio
    async def test_cxcywh_without_image_dims(
        self,
        converter: GeometryConverter,
    ) -> None:
        result = await converter.to_canonical_bbox(
            (100, 200, 50, 80),
            "cxcywh",
        )
        assert result == (75.0, 160.0, 50.0, 80.0)

    @pytest.mark.asyncio
    async def test_normalized_xywh(
        self,
        converter: GeometryConverter,
    ) -> None:
        result = await converter.to_canonical_bbox(
            (0.1, 0.2, 0.5, 0.4),
            "normalized_xywh",
            image_width=640,
            image_height=480,
        )
        assert result == (64.0, 96.0, 320.0, 192.0)

    @pytest.mark.asyncio
    async def test_normalized_missing_dims(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.to_canonical_bbox(
                (0.1, 0.2, 0.5, 0.4),
                "normalized_xywh",
            )

    @pytest.mark.asyncio
    async def test_unknown_format(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.to_canonical_bbox(
                (1, 2, 3, 4),
                "unknown_format",
            )

    @pytest.mark.asyncio
    async def test_invalid_xyxy_reversed(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.to_canonical_bbox((100, 100, 50, 50), "xyxy")

    @pytest.mark.asyncio
    async def test_invalid_negative_dims(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.to_canonical_bbox((10, 10, -5, 10), "xywh")

    @pytest.mark.asyncio
    async def test_normalize(self, converter: GeometryConverter) -> None:
        result = await converter.normalize(100, 200, 50, 80, 640, 480)
        assert result == pytest.approx((0.15625, 0.41667, 0.078125, 0.16667), rel=0.001)

    @pytest.mark.asyncio
    async def test_normalize_invalid_dims(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.normalize(0, 0, 10, 10, 0, 480)

    @pytest.mark.asyncio
    async def test_denormalize(self, converter: GeometryConverter) -> None:
        result = await converter.denormalize(0.1, 0.2, 0.5, 0.3, 640, 480)
        assert result == (64.0, 96.0, 320.0, 144.0)

    @pytest.mark.asyncio
    async def test_polygon_to_bbox(self, converter: GeometryConverter) -> None:
        result = await converter.polygon_to_bbox(
            [(10, 20), (110, 20), (110, 220), (10, 220)],
        )
        assert result == (10, 20, 100, 200)

    @pytest.mark.asyncio
    async def test_polygon_to_bbox_insufficient_vertices(
        self,
        converter: GeometryConverter,
    ) -> None:
        with pytest.raises(GeometryError):
            await converter.polygon_to_bbox([(0, 0), (1, 1)])

    @pytest.mark.asyncio
    async def test_obb_to_bbox(self, converter: GeometryConverter) -> None:
        result = await converter.obb_to_bbox(100, 100, 50, 30, 0)
        assert result == (75.0, 85.0, 50.0, 30.0)

    @pytest.mark.asyncio
    async def test_obb_to_bbox_rotated(self, converter: GeometryConverter) -> None:
        result = await converter.obb_to_bbox(100, 100, 50, 30, 45)
        w, h = result[2], result[3]
        assert w > 50
        assert h > 30

    @pytest.mark.asyncio
    async def test_bbox_to_cxcywh(self, converter: GeometryConverter) -> None:
        result = await converter.bbox_to_cxcywh(10, 20, 100, 200)
        assert result == (60.0, 120.0, 100.0, 200.0)

    @pytest.mark.asyncio
    async def test_xyxy_zero_dimensions(self, converter: GeometryConverter) -> None:
        result = await converter.to_canonical_bbox((10, 20, 10, 20), "xyxy")
        assert result == (10, 20, 0, 0)

    @pytest.mark.asyncio
    async def test_cxcywh_zero_dimensions(self, converter: GeometryConverter) -> None:
        result = await converter.to_canonical_bbox((100, 100, 0, 0), "cxcywh")
        assert result == (100.0, 100.0, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_obb_zero_angle(self, converter: GeometryConverter) -> None:
        result = await converter.obb_to_bbox(50, 50, 100, 60, 0)
        assert result == (0.0, 20.0, 100.0, 60.0)

    @pytest.mark.asyncio
    async def test_obb_negative_angle(self, converter: GeometryConverter) -> None:
        result = await converter.obb_to_bbox(50, 50, 100, 60, -90)
        w, h = result[2], result[3]
        assert w > 0 and h > 0

    @pytest.mark.asyncio
    async def test_polygon_to_bbox_triangle(self, converter: GeometryConverter) -> None:
        result = await converter.polygon_to_bbox([(0, 0), (100, 0), (50, 100)])
        assert result == (0, 0, 100, 100)

    @pytest.mark.asyncio
    async def test_normalize_edge_values(self, converter: GeometryConverter) -> None:
        result = await converter.normalize(0, 0, 640, 480, 640, 480)
        assert result == (0.0, 0.0, 1.0, 1.0)

    @pytest.mark.asyncio
    async def test_denormalize_edge_values(self, converter: GeometryConverter) -> None:
        result = await converter.denormalize(1.0, 1.0, 1.0, 1.0, 640, 480)
        assert result == (640.0, 480.0, 640.0, 480.0)

    @pytest.mark.asyncio
    async def test_to_canonical_bbox_wrong_length(self, converter: GeometryConverter) -> None:
        with pytest.raises(GeometryError, match="Expected 4 values"):
            await converter.to_canonical_bbox((1, 2, 3), "xyxy")

    @pytest.mark.asyncio
    async def test_xyxy_xmax_equals_xmin(self, converter: GeometryConverter) -> None:
        result = await converter.to_canonical_bbox((10, 20, 10, 100), "xyxy")
        assert result == (10, 20, 0, 80)
