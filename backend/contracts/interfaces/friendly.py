"""Interface for the Friendly Intelligence Agent."""

from abc import ABC, abstractmethod

from backend.contracts.models.analysis import FriendlyAnalysis
from backend.contracts.models.detection import DetectionResult


class FriendlyModule(ABC):
    """Contract for analysing whether detected subjects are friendly forces."""

    @abstractmethod
    async def analyze_friendly(self, detection: DetectionResult) -> FriendlyAnalysis:
        """Assess if any objects in the detection result match known friendly units.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        FriendlyAnalysis
            Assessment of friendly-force matches and confidence.
        """
        ...
