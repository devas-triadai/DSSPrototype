"""Interface for the Enemy Intelligence Agent."""

from abc import ABC, abstractmethod

from backend.contracts.models.analysis import EnemyAnalysis
from backend.contracts.models.detection import DetectionResult


class EnemyModule(ABC):
    """Contract for analysing whether detected subjects are enemy forces."""

    @abstractmethod
    async def analyze_enemy(self, detection: DetectionResult) -> EnemyAnalysis:
        """Assess if any objects in the detection result match known enemy units.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        EnemyAnalysis
            Assessment of enemy-force matches, possible equipment, and confidence.
        """
        ...
