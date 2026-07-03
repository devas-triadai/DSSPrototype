"""Spatial geometry models for object detection.

Supports three annotation styles:
  - Axis-aligned bounding box (BoundingBox)
  - Oriented bounding box (OrientedBBox) — rotated rectangle
  - Polygon segmentation (Polygon) — arbitrary vertex list
"""

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Pixel-coordinate axis-aligned rectangle.

    The (x, y) origin is the top-left corner of the image.
    Width and height must be strictly positive.
    This is the DEFAULT annotation style (lowest cost).
    """

    model_config = ConfigDict(frozen=True)

    x: float = Field(..., ge=0, description="X-coordinate of the top-left corner")
    y: float = Field(..., ge=0, description="Y-coordinate of the top-left corner")
    width: float = Field(..., gt=0, description="Width of the bounding box in pixels")
    height: float = Field(..., gt=0, description="Height of the bounding box in pixels")


class OrientedBBox(BaseModel):
    """Rotated rectangle defined by center, size, and rotation angle.

    Use for vehicles, ships, and other objects with a clear heading.
    Angle is measured clockwise from the positive x-axis in degrees.
    """

    model_config = ConfigDict(frozen=True)

    cx: float = Field(..., ge=0, description="X-coordinate of the center")
    cy: float = Field(..., ge=0, description="Y-coordinate of the center")
    width: float = Field(..., gt=0, description="Width of the bounding box (long axis)")
    height: float = Field(..., gt=0, description="Height of the bounding box (short axis)")
    angle: float = Field(
        ..., description="Rotation angle in degrees clockwise from x-axis"
    )


class Polygon(BaseModel):
    """Closed polygon defined by a list of (x, y) vertices.

    The first and last vertices are assumed to connect.
    Minimum 3 vertices required (triangle).
    """

    model_config = ConfigDict(frozen=True)

    vertices: list[tuple[float, float]] = Field(
        ..., min_length=3, description="List of (x, y) vertex coordinates"
    )


class AnnotationGeometry(BaseModel):
    """Container holding all geometry formats for a single detection.

    At minimum, `box` must be present. `obb` and `polygon` are optional
    enhancements. Downstream modules should prefer more precise geometry
    when available.
    """

    model_config = ConfigDict(frozen=True)

    box: BoundingBox = Field(..., description="Axis-aligned bounding box (always present)")
    obb: OrientedBBox | None = Field(None, description="Oriented bounding box (optional)")
    polygon: Polygon | None = Field(None, description="Segmentation polygon (optional)")
    segmentation_mask: str | None = Field(
        None, description="Base64-encoded RLE or PNG segmentation mask (optional)"
    )
