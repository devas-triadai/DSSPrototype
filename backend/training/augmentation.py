"""Augmentation pipeline — configurable transforms for CV training.

Supports: Flip, Rotate, Scale, Crop, Brightness, Contrast, Blur,
Noise, Mosaic, MixUp. Framework-agnostic configuration; framework-
specific implementations are injected via TrainingBackendInterface.
"""

import logging
from dataclasses import dataclass

from backend.training.interfaces import AugmentationPipelineInterface
from backend.training.models import AugmentationConfig

logger = logging.getLogger("dss.training.augmentation")


@dataclass(frozen=True)
class TransformDescriptor:
    """Describes an available augmentation transform."""

    name: str
    description: str
    configurable_params: tuple[str, ...]


_AVAILABLE_TRANSFORMS: tuple[TransformDescriptor, ...] = (
    TransformDescriptor("flip", "Horizontal and vertical flip", ("flip_probability",)),
    TransformDescriptor(
        "rotate", "Random rotation within degree range", ("rotate_degrees", "rotate_probability"),
    ),
    TransformDescriptor(
        "scale", "Random scaling within min/max range",
        ("scale_min", "scale_max", "scale_probability"),
    ),
    TransformDescriptor(
        "crop", "Random crop within min/max range", ("crop_min", "crop_max", "crop_probability"),
    ),
    TransformDescriptor(
        "brightness", "Brightness adjustment", ("brightness_range", "brightness_probability"),
    ),
    TransformDescriptor(
        "contrast", "Contrast adjustment", ("contrast_range", "contrast_probability"),
    ),
    TransformDescriptor("blur", "Gaussian blur", ("blur_kernel_size", "blur_probability")),
    TransformDescriptor(
        "noise", "Random noise injection", ("noise_intensity", "noise_probability"),
    ),
    TransformDescriptor("mosaic", "Mosaic augmentation", ("mosaic_probability",)),
    TransformDescriptor("mixup", "MixUp augmentation", ("mixup_probability",)),
)


class AugmentationPipeline(AugmentationPipelineInterface):
    """Framework-agnostic augmentation pipeline configuration.

    Stores augmentation configurations and returns pipeline descriptors.
    Actual transform execution is delegated to framework-specific backends.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}

    def create_pipeline(self, config: AugmentationConfig) -> dict[str, object]:
        """Create an augmentation pipeline configuration dict.

        Returns a serializable dict that framework-specific backends
        can consume to build their native augmentation pipelines.
        """
        logger.info("Creating augmentation pipeline: %s", config.name)
        pipeline: dict[str, object] = {
            "name": config.name,
            "transforms": [],
        }

        transforms: list[dict[str, object]] = pipeline["transforms"]  # type: ignore[assignment]

        if config.flip_probability > 0:
            transforms.append({"type": "flip", "probability": config.flip_probability})

        if config.rotate_probability > 0:
            transforms.append({
                "type": "rotate",
                "degrees": list(config.rotate_degrees),
                "probability": config.rotate_probability,
            })

        if config.scale_probability > 0:
            transforms.append({
                "type": "scale",
                "min": config.scale_min,
                "max": config.scale_max,
                "probability": config.scale_probability,
            })

        if config.crop_probability > 0:
            transforms.append({
                "type": "crop",
                "min": config.crop_min,
                "max": config.crop_max,
                "probability": config.crop_probability,
            })

        if config.brightness_probability > 0:
            transforms.append({
                "type": "brightness",
                "range": list(config.brightness_range),
                "probability": config.brightness_probability,
            })

        if config.contrast_probability > 0:
            transforms.append({
                "type": "contrast",
                "range": list(config.contrast_range),
                "probability": config.contrast_probability,
            })

        if config.blur_probability > 0:
            transforms.append({
                "type": "blur",
                "kernel_size": config.blur_kernel_size,
                "probability": config.blur_probability,
            })

        if config.noise_probability > 0:
            transforms.append({
                "type": "noise",
                "intensity": config.noise_intensity,
                "probability": config.noise_probability,
            })

        if config.mosaic_probability > 0:
            transforms.append({
                "type": "mosaic",
                "probability": config.mosaic_probability,
            })

        if config.mixup_probability > 0:
            transforms.append({
                "type": "mixup",
                "probability": config.mixup_probability,
            })

        self._pipelines[config.name] = pipeline
        logger.info(
            "Augmentation pipeline created: %s (%d transforms)", config.name, len(transforms),
        )
        return pipeline

    def get_available_transforms(self) -> list[str]:
        return [t.name for t in _AVAILABLE_TRANSFORMS]
