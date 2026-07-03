"""Computer Vision module configuration.

All values are overridable via environment variables prefixed with ``CV_``.
"""

from pydantic_settings import BaseSettings


class CVConfig(BaseSettings):
    """Configuration for the Computer Vision pipeline.

    Controls model selection, inference parameters, image constraints,
    and hardware allocation.
    """

    model_config = {"env_prefix": "CV_"}

    model_path: str = "models/yolov8n.pt"
    default_model_type: str = "yolo"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    max_image_width: int = 1920
    max_image_height: int = 1080
    supported_formats: list[str] = ["jpg", "jpeg", "png", "tiff", "bmp", "webp"]
    max_file_size_mb: int = 50


cv_config = CVConfig()
