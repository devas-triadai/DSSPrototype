"""Converts raw model output to strongly typed contract models.

Maps raw bounding boxes, class IDs, and confidence scores
to the ``DetectionResult`` structure defined in ``backend.contracts``.
CV output is PURE PERCEPTION — no semantic enrichment.
"""

from datetime import datetime, timezone

from backend.contracts.enums.core import ObjectType
from backend.contracts.models.detection import DetectedObject, DetectionResult, ImageMetadata
from backend.contracts.models.geometry import AnnotationGeometry, BoundingBox
from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.interfaces import RawInferenceOutput, ResultConverterInterface


class ResultConverter(ResultConverterInterface):
    """Convert raw detections into ``DetectionResult`` contract models.

    A class-to-type mapping must be provided. The mapper translates
    model-specific class IDs into the ontology ``ObjectType`` enum.
    ``class_name_mapping`` provides a fallback keyed by the model's
    human-readable class name (case-insensitive, underscores/spaces
    normalised) so that custom military models work even when their
    class IDs differ from COCO.
    """

    def __init__(
        self,
        class_mapping: dict[int, ObjectType] | None = None,
        class_name_mapping: dict[str, ObjectType] | None = None,
    ) -> None:
        self._class_mapping = class_mapping or {}
        self._class_name_mapping = class_name_mapping or {}
        self._threshold = cv_config.confidence_threshold

    def convert(
        self,
        raw: RawInferenceOutput,
        metadata: ImageMetadata,
        model_version: str,
    ) -> DetectionResult:
        """Map raw detections to a standard ``DetectionResult``.

        Each detection becomes a ``DetectedObject`` carrying only
        perception data: type, confidence, and geometry.

        Parameters
        ----------
        raw:
            Raw inference output from the model.
        metadata:
            Metadata for the source image.
        model_version:
            Version string to attach to the result.

        Returns
        -------
        DetectionResult
            Strongly typed result with ontology-mapped classes.
        """
        objects: list[DetectedObject] = []

        for detection in raw.detections:
            if detection.confidence < self._threshold:
                continue

            object_type = self._class_mapping.get(detection.class_id)
            if object_type is None:
                normalized = detection.class_name.strip().lower().replace(" ", "_").replace("-", "_")
                object_type = self._class_name_mapping.get(normalized, ObjectType.UNKNOWN_OBJECT)

            geometry = AnnotationGeometry(
                box=BoundingBox(
                    x=detection.bbox.x1,
                    y=detection.bbox.y1,
                    width=detection.bbox.x2 - detection.bbox.x1,
                    height=detection.bbox.y2 - detection.bbox.y1,
                ),
            )

            objects.append(
                DetectedObject(
                    name=detection.class_name,
                    object_type=object_type,
                    confidence=detection.confidence,
                    geometry=geometry,
                )
            )

        return DetectionResult(
            image_id=metadata.image_id,
            timestamp=datetime.now(timezone.utc),
            objects=objects,
            model_version=model_version,
            processing_time_ms=raw.processing_time_ms,
        )
