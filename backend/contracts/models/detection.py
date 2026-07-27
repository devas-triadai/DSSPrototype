"""Models representing computer-vision detection outputs.

CV output contains ONLY perception data — physical object types,
confidence scores, and spatial geometry. No semantic annotations.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.contracts.enums.core import ObjectType
from backend.contracts.models.geometry import AnnotationGeometry


class ImageMetadata(BaseModel):
    """Metadata describing a source image submitted for analysis."""

    model_config = ConfigDict(frozen=True)

    image_id: str = Field(..., description="Unique identifier for the image")
    timestamp: datetime = Field(..., description="When the image was captured")
    source: str | None = Field(None, description="Source or sensor identifier")
    width: int | None = Field(None, ge=1, description="Image width in pixels")
    height: int | None = Field(None, ge=1, description="Image height in pixels")
    format: str | None = Field(None, description="Image file format, e.g. JPEG or PNG")


class DetectedObject(BaseModel):
    """A single physical object identified within an image.

    This is a PURE PERCEPTION output. It describes WHAT exists
    and WHERE — not what it means. Semantic enrichment (friendly,
    enemy, threat, intent) is applied by Knowledge Modules.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique detection identifier",
    )
    name: str = Field(
        ..., description="Model class name, e.g. K9_Vajra, Baktar_Shikan_ATGM"
    )
    object_type: ObjectType = Field(
        default=ObjectType.UNKNOWN_OBJECT,
        description="Perception-only class from the CV ontology",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score in [0, 1]"
    )
    geometry: AnnotationGeometry = Field(
        ..., description="Spatial location(s) within the image"
    )


class DetectionResult(BaseModel):
    """Complete output of a single image-detection pass.

    This is the IMMUTABLE contract between CV and all downstream
    modules. Every field is frozen and versioned.
    """

    model_config = ConfigDict(frozen=True)

    image_id: str = Field(..., description="Foreign key referencing the source image")
    timestamp: datetime = Field(..., description="When this detection result was produced")
    objects: list[DetectedObject] = Field(..., description="All objects detected in the image")
    model_version: str = Field(..., description="Version identifier of the detection model")
    processing_time_ms: float = Field(
        ..., ge=0, description="Inference duration in milliseconds"
    )
