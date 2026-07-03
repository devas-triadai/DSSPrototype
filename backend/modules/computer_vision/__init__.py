"""Computer Vision module — perception layer.

Ingests raw imagery and produces structured ``DetectionResult`` payloads
via a model-agnostic pipeline.  New vision models are added through the
plugin registry in ``model_manager``.
"""

from backend.modules.computer_vision.config import CVConfig, cv_config
from backend.modules.computer_vision.exceptions import (
    ImageValidationError,
    InferenceError,
    ModelLoadError,
    PreprocessingError,
)
from backend.modules.computer_vision.image_loader import ImageLoader
from backend.modules.computer_vision.image_preprocessor import ImagePreprocessor
from backend.modules.computer_vision.image_validator import ImageValidator
from backend.modules.computer_vision.inference_engine import InferenceEngine
from backend.modules.computer_vision.interfaces import (
    ImageLoaderInterface,
    ImagePreprocessorInterface,
    ImageValidatorInterface,
    InferenceEngineInterface,
    ModelManagerInterface,
    ModelMetadata,
    RawBoundingBox,
    RawDetection,
    RawInferenceOutput,
    ResultConverterInterface,
    VisionModelInterface,
)
from backend.modules.computer_vision.model_manager import ModelManager, register_model
from backend.modules.computer_vision.result_converter import ResultConverter
from backend.modules.computer_vision.service import ComputerVisionService
from backend.modules.computer_vision.yolo_adapter import YOLOModel

__all__ = [
    "CVConfig",
    "cv_config",
    "ComputerVisionService",
    "ModelManager",
    "register_model",
    "ImageLoader",
    "ImageValidator",
    "ImagePreprocessor",
    "InferenceEngine",
    "ResultConverter",
    "ModelLoadError",
    "InferenceError",
    "ImageValidationError",
    "PreprocessingError",
    "VisionModelInterface",
    "ImageLoaderInterface",
    "ImageValidatorInterface",
    "ImagePreprocessorInterface",
    "InferenceEngineInterface",
    "ModelManagerInterface",
    "ResultConverterInterface",
    "YOLOModel",
    "ModelMetadata",
    "RawBoundingBox",
    "RawDetection",
    "RawInferenceOutput",
]
