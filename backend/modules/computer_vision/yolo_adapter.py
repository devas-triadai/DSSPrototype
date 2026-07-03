"""Ultralytics YOLO model adapter.

Implements ``VisionModelInterface`` to wrap Ultralytics YOLO models
(YOLOv8, YOLOv9, YOLOv10, YOLO11, etc.) as a plugin for the
model-agnostic Computer Vision pipeline.

Auto-registers itself with ``ModelManager`` at import time so
it is discoverable via ``load_model("yolo", ...)``.
"""

import logging
from typing import Any

import numpy as np

from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.interfaces import (
    ModelMetadata,
    RawBoundingBox,
    RawDetection,
    RawInferenceOutput,
    VisionModelInterface,
)
from backend.modules.computer_vision.model_manager import register_model

logger = logging.getLogger("dss.computer_vision.yolo_adapter")


class YOLOModel(VisionModelInterface):
    """Adapter that wraps an Ultralytics YOLO model.

    Configuration is read from ``cv_config`` at runtime:

    * **model_path** — path to the ``.pt`` weights file
    * **device** — target device (``"cpu"``, ``"cuda:0"``, …)
    * **confidence_threshold** — minimum confidence to keep a detection
    * **iou_threshold** — IoU threshold for NMS
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._config = cv_config
        self._status: str = "unloaded"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load YOLO weights and move the model to the configured device.

        Raises
        ------
        ModelLoadError (wrapped by ``ModelManager``)
            If Ultralytics is not installed or the weights cannot be
            loaded.
        """
        from ultralytics import YOLO

        model_path = self._config.model_path
        device = self._config.device

        logger.info("Loading YOLO model from '%s' on device '%s'", model_path, device)

        self._status = "loading"
        try:
            self._model = YOLO(model_path)
            self._model.to(device)
            self._status = "loaded"
            logger.info("YOLO model loaded: %s", model_path)
        except Exception:
            self._status = "error"
            raise

    def unload(self) -> None:
        """Release the model and free resources."""
        self._model = None
        self._status = "unloaded"
        logger.info("YOLO model unloaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> RawInferenceOutput:
        """Run YOLO inference on a pre-processed image.

        Parameters
        ----------
        image:
            RGB image array (H, W, 3).

        Returns
        -------
        RawInferenceOutput
            Raw detections with bounding boxes, confidence scores,
            and class labels.
        """
        if self._model is None:
            raise RuntimeError("YOLO model not loaded. Call load() first.")

        results = self._model(
            image,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            device=self._config.device,
            verbose=False,
        )

        detections: list[RawDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.xyxy.shape[0] == 0:
                continue

            for i in range(boxes.xyxy.shape[0]):
                xyxy = boxes.xyxy[i].tolist()
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = str(result.names[class_id]) if result.names else str(class_id)

                detections.append(
                    RawDetection(
                        bbox=RawBoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return RawInferenceOutput(detections=detections)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> ModelMetadata:
        name = str(self._config.model_path)
        version = "unknown"
        if self._model is not None:
            try:
                ckpt = getattr(self._model, "ckpt", None)
                if ckpt is not None:
                    version = str(ckpt.get("version", "unknown"))
            except Exception:
                pass
        return ModelMetadata(
            name=name,
            version=version,
            model_type="yolo",
            device=self._config.device,
            input_size=None,
        )

    @property
    def status(self) -> str:
        return self._status


# -----------------------------------------------------------------------
# Plugin auto-registration — makes "yolo" discoverable by ModelManager
# -----------------------------------------------------------------------

register_model("yolo", YOLOModel)
