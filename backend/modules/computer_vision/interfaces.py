"""Abstract interfaces for every component in the Computer Vision pipeline.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from backend.contracts.models.detection import DetectionResult, ImageMetadata

# ---------------------------------------------------------------------------
# Raw (pre-contract) inference types — internal to this module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawBoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates.

    (x1, y1) is the top-left corner, (x2, y2) the bottom-right.
    """

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class RawDetection:
    """A single detection produced by a model before contract conversion."""

    bbox: RawBoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class RawInferenceOutput:
    """Complete raw output from one inference pass."""

    detections: list[RawDetection] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

DEFAULT_INPUT_SIZE: tuple[int, int] | None = None


@dataclass(frozen=True)
class ModelMetadata:
    """Read-only descriptor for a loaded model instance."""

    name: str = ""
    version: str = ""
    model_type: str = ""
    device: str = "cpu"
    input_size: tuple[int, int] | None = DEFAULT_INPUT_SIZE


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class VisionModelInterface(ABC):
    """Contract for any object-detection model.

    Implementations wrap model-specific frameworks (YOLO, RT-DETR, etc.)
    and expose a uniform interface.
    """

    @abstractmethod
    def load(self) -> None:
        """Allocate model resources (weights, device transfer)."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> RawInferenceOutput:
        """Run inference on a pre-processed image array.

        Parameters
        ----------
        image:
            Pre-processed image as a NumPy array (H, W, C).

        Returns
        -------
        RawInferenceOutput
            Raw detections and timing information.
        """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return read-only metadata for this model instance."""

    @property
    @abstractmethod
    def status(self) -> str:
        """Return current model state: unloaded | loading | loaded | error."""


class ImageLoaderInterface(ABC):
    """Contract for loading images from various sources."""

    @abstractmethod
    def load(self, source: str | bytes) -> np.ndarray:
        """Load an image and return it as a NumPy array (H, W, C).

        Parameters
        ----------
        source:
            A filesystem path or raw image bytes.

        Returns
        -------
        np.ndarray
            Decoded image in RGB order.
        """


class ImageValidatorInterface(ABC):
    """Contract for validating image properties before processing."""

    @abstractmethod
    def validate(self, image: np.ndarray) -> None:
        """Raise ``ImageValidationError`` if the image fails any check.

        Parameters
        ----------
        image:
            The image array to validate.
        """


Transform = Callable[[np.ndarray], np.ndarray]


class ImagePreprocessorInterface(ABC):
    """Contract for preparing images for model inference."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply the preprocessing pipeline to *image*.

        Parameters
        ----------
        image:
            Raw loaded image (H, W, C).

        Returns
        -------
        np.ndarray
            Pre-processed image ready for inference.
        """


class InferenceEngineInterface(ABC):
    """Contract for executing model inference."""

    @abstractmethod
    def run(self, model: VisionModelInterface, image: np.ndarray) -> RawInferenceOutput:
        """Run *model* on *image* and return raw detections.

        Parameters
        ----------
        model:
            A loaded model instance.
        image:
            Pre-processed image array.

        Returns
        -------
        RawInferenceOutput
            Raw detections and timing.
        """


class ResultConverterInterface(ABC):
    """Contract for converting raw model output to contract models."""

    @abstractmethod
    def convert(
        self,
        raw: RawInferenceOutput,
        metadata: ImageMetadata,
        model_version: str,
    ) -> DetectionResult:
        """Map raw detections to the standard ``DetectionResult`` contract.

        Parameters
        ----------
        raw:
            Raw inference output from the model.
        metadata:
            Source image metadata.
        model_version:
            Version string to embed in the result.

        Returns
        -------
        DetectionResult
            Strongly typed result conforming to ``backend.contracts``.
        """


class ModelManagerInterface(ABC):
    """Contract for managing model lifecycle and the model registry."""

    @abstractmethod
    def load_model(self, model_type: str, model_path: str) -> VisionModelInterface:
        """Load a model of *model_type* from *model_path*.

        Returns
        -------
        VisionModelInterface
            The loaded model instance.
        """

    @abstractmethod
    def get_model(self, model_type: str) -> VisionModelInterface:
        """Retrieve a previously loaded model by type."""

    @abstractmethod
    def unload_model(self, model_type: str) -> None:
        """Unload a model and free its resources."""

    @abstractmethod
    def reload_model(self, model_type: str) -> None:
        """Reload a model (unload then load with current configuration)."""

    @abstractmethod
    def list_models(self) -> list[ModelMetadata]:
        """Return metadata for all currently loaded models."""
