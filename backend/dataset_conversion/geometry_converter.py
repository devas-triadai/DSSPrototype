from __future__ import annotations

import math
from collections.abc import Sequence

from backend.dataset_conversion.exceptions import GeometryError
from backend.dataset_conversion.interfaces import GeometryConverterInterface


class GeometryConverter(GeometryConverterInterface):
    async def to_canonical_bbox(
        self,
        geometry: Sequence[float],
        source_format: str,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> tuple[float, float, float, float]:
        if len(geometry) != 4:
            raise GeometryError(f"Expected 4 values, got {len(geometry)}")

        x1, y1, x2_or_w, y2_or_h = geometry

        if source_format == "xyxy":
            if x2_or_w < x1:
                raise GeometryError(f"Invalid bbox: xmax ({x2_or_w}) < xmin ({x1})")
            if y2_or_h < y1:
                raise GeometryError(f"Invalid bbox: ymax ({y2_or_h}) < ymin ({y1})")
            return (x1, y1, x2_or_w - x1, y2_or_h - y1)
        elif source_format == "xywh":
            if x2_or_w < 0 or y2_or_h < 0:
                raise GeometryError(f"Invalid bbox: negative dimensions ({x2_or_w}, {y2_or_h})")
            return (x1, y1, x2_or_w, y2_or_h)
        elif source_format == "cxcywh":
            cx, cy, w, h = geometry
            if w < 0 or h < 0:
                raise GeometryError(f"Invalid bbox: negative dimensions ({w}, {h})")
            if image_width and image_height:
                cx_abs = cx * image_width
                cy_abs = cy * image_height
                w_abs = w * image_width
                h_abs = h * image_height
            else:
                cx_abs, cy_abs, w_abs, h_abs = cx, cy, w, h
            return (cx_abs - w_abs / 2, cy_abs - h_abs / 2, w_abs, h_abs)
        elif source_format == "normalized_xywh":
            if not image_width or not image_height:
                raise GeometryError(
                    "image_width and image_height required for normalized coordinates"
                )
            return (
                x1 * image_width,
                y1 * image_height,
                x2_or_w * image_width,
                y2_or_h * image_height,
            )
        else:
            raise GeometryError(f"Unknown source format: {source_format}")

    async def normalize(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            raise GeometryError(f"Invalid image dimensions: {image_width}x{image_height}")
        return (
            x / image_width,
            y / image_height,
            width / image_width,
            height / image_height,
        )

    async def denormalize(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            raise GeometryError(f"Invalid image dimensions: {image_width}x{image_height}")
        return (
            x * image_width,
            y * image_height,
            width * image_width,
            height * image_height,
        )

    async def polygon_to_bbox(
        self,
        vertices: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        if len(vertices) < 3:
            raise GeometryError(f"Polygon must have at least 3 vertices, got {len(vertices)}")
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    async def obb_to_bbox(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        angle: float,
    ) -> tuple[float, float, float, float]:
        angle_rad = math.radians(angle)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        w_rot = width * cos_a + height * sin_a
        h_rot = width * sin_a + height * cos_a
        x = cx - w_rot / 2
        y = cy - h_rot / 2
        return (x, y, w_rot, h_rot)

    async def bbox_to_cxcywh(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[float, float, float, float]:
        return (x + width / 2, y + height / 2, width, height)
