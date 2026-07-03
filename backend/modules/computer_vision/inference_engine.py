"""Inference execution layer.

Delegates to the model and adds timing instrumentation.
Contains no model-specific logic.
"""

import time

import numpy as np

from backend.modules.computer_vision.exceptions import InferenceError
from backend.modules.computer_vision.interfaces import (
    InferenceEngineInterface,
    RawInferenceOutput,
    VisionModelInterface,
)


class InferenceEngine(InferenceEngineInterface):
    """Executes model inference with timing and error handling."""

    def run(self, model: VisionModelInterface, image: np.ndarray) -> RawInferenceOutput:
        """Run *model* on *image* and return raw detections.

        Parameters
        ----------
        model:
            A loaded model instance.
        image:
            Pre-processed image array ready for inference.

        Returns
        -------
        RawInferenceOutput
            Raw detections and wall-clock inference time.

        Raises
        ------
        InferenceError
            If the model call fails.
        """
        try:
            start = time.perf_counter()
            raw_output = model.predict(image)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            return RawInferenceOutput(
                detections=raw_output.detections,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
