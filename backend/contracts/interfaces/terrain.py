"""Interface for the Terrain Intelligence Agent."""

from abc import ABC, abstractmethod

from backend.contracts.models.analysis import TerrainAnalysis
from backend.contracts.models.detection import DetectionResult


class TerrainModule(ABC):
    """Contract for analysing terrain characteristics from detection data."""

    @abstractmethod
    async def analyze_terrain(self, detection: DetectionResult) -> TerrainAnalysis:
        """Analyse terrain features, visibility, and accessibility.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        TerrainAnalysis
            Terrain classification, features, and mobility assessment.
        """
        ...
