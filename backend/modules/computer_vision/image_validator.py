"""Image validation service.

Checks format, dimensions, and basic integrity before processing.
"""

import numpy as np

from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.exceptions import ImageValidationError
from backend.modules.computer_vision.interfaces import ImageValidatorInterface


class ImageValidator(ImageValidatorInterface):
    """Validates image properties against configured constraints."""

    def __init__(self) -> None:
        self._formats = cv_config.supported_formats
        self._max_width = cv_config.max_image_width
        self._max_height = cv_config.max_image_height

    def validate(self, image: np.ndarray) -> None:
        """Raise ``ImageValidationError`` if the image fails any check.

        Parameters
        ----------
        image:
            Image array in (H, W, C) format.

        Raises
        ------
        ImageValidationError
            On any validation failure.
        """
        if image is None or image.size == 0:
            raise ImageValidationError("Image is empty")

        if len(image.shape) < 2:
            raise ImageValidationError(f"Invalid image dimensions: {image.shape}")

        height, width = image.shape[:2]

        if width > self._max_width or height > self._max_height:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) exceed limit "
                f"({self._max_width}x{self._max_height})"
            )
