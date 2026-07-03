"""Reusable image preprocessing pipeline.

Transforms are composed at construction time and applied sequentially.
Model-specific preprocessing is injected via the ``transforms`` parameter.
"""

from collections.abc import Callable

import numpy as np

from backend.modules.computer_vision.interfaces import (
    ImagePreprocessorInterface,
    Transform,
)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure the image has exactly three channels (RGB)."""
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize the image to the target dimensions."""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(image)
    resized = pil.resize((width, height), PILImage.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def make_resize(width: int, height: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a resize transform configured for *width* x *height*."""

    def _resize(image: np.ndarray) -> np.ndarray:
        return resize(image, width, height)

    return _resize


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Convert pixel values from [0, 255] to [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0


class ImagePreprocessor(ImagePreprocessorInterface):
    """Configurable preprocessing pipeline.

    By default applies only RGB conversion.  Extend by providing
    custom transforms::

        preprocessor = ImagePreprocessor([
            to_rgb,
            make_resize(640, 640),
            normalize_uint8,
        ])
    """

    def __init__(self, transforms: list[Transform] | None = None) -> None:
        self._transforms = transforms if transforms is not None else [to_rgb]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply all configured transforms in order.

        Parameters
        ----------
        image:
            Raw loaded image (H, W, C).

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        result = image
        for transform in self._transforms:
            result = transform(result)
        return result
