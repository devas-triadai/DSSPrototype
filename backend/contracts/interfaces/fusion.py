"""Interface for the Fusion Agent."""

from abc import ABC, abstractmethod

from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.contracts.models.fusion import FusionResult, ThreatAssessment


class FusionModule(ABC):
    """Contract for fusing multi-source intelligence into a unified picture."""

    @abstractmethod
    async def fuse_intelligence(
        self,
        friendly: FriendlyAnalysis,
        enemy: EnemyAnalysis,
        terrain: TerrainAnalysis,
    ) -> FusionResult:
        """Merge analyses from all three domain agents into a single fused picture.

        Parameters
        ----------
        friendly:
            Assessment from the Friendly Intelligence Agent.
        enemy:
            Assessment from the Enemy Intelligence Agent.
        terrain:
            Assessment from the Terrain Intelligence Agent.

        Returns
        -------
        FusionResult
            Combined confidence, summary, and supporting evidence.
        """
        ...

    @abstractmethod
    async def assess_threat(self, fusion: FusionResult) -> ThreatAssessment:
        """Derive a threat-level assessment from the fused intelligence.

        Parameters
        ----------
        fusion:
            The fused intelligence picture.

        Returns
        -------
        ThreatAssessment
            Severity level, confidence, and reasoning.
        """
        ...
