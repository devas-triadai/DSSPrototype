"""Interface for the Computer Vision module."""

from abc import ABC, abstractmethod

from backend.contracts.models.detection import DetectionResult, ImageMetadata


class VisionModule(ABC):
    """Contract for processing imagery and detecting objects.

    Every Computer Vision implementation must satisfy this interface.
    """

    @abstractmethod
    async def process_image(self, image: ImageMetadata) -> DetectionResult:
        """Run detection on a single image and return all identified objects.

        Parameters
        ----------
        image:
            Metadata describing the source image to process.

        Returns
        -------
        DetectionResult
            All objects detected in the image with confidence scores.
        """
        ...
