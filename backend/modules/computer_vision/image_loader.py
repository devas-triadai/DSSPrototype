"""Image loading from filesystem paths or raw bytes.

Uses Pillow for safe decoding and converts to NumPy arrays.
"""

from io import BytesIO

import numpy as np
from PIL import Image

from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.exceptions import ImageValidationError
from backend.modules.computer_vision.interfaces import ImageLoaderInterface


class ImageLoader(ImageLoaderInterface):
    """Load images from local paths or byte buffers."""

    def __init__(self) -> None:
        self._formats = cv_config.supported_formats

    def load(self, source: str | bytes) -> np.ndarray:
        """Decode an image and return it as an RGB NumPy array.

        Parameters
        ----------
        source:
            A filesystem path (``str``) or raw image bytes.

        Returns
        -------
        np.ndarray
            RGB image array of shape (H, W, 3).

        Raises
        ------
        ImageValidationError
            If the source cannot be decoded as a supported image format.
        """
        try:
            if isinstance(source, str):
                pil_image: Image.Image = Image.open(source)
            else:
                pil_image = Image.open(BytesIO(source))

            pil_image.load()
            pil_image = pil_image.convert("RGB")
            return np.asarray(pil_image, dtype=np.uint8)

        except (OSError, ValueError, MemoryError) as exc:
            raise ImageValidationError(
                f"Failed to load image from provided source: {exc}"
            ) from exc
