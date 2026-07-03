"""Intelligence collection for the Fusion Engine.

Aggregates analyses from all three knowledge modules into a single
structure.  Performs no validation, correlation, or confidence
calculations.
"""

from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    CollectorInterface,
)


class Collector(CollectorInterface):
    """Collects intelligence from all three domain modules.

    Simply packages the three analyses into a ``CollectedIntelligence``
    dataclass for downstream processing.
    """

    def collect(
        self,
        friendly: FriendlyAnalysis,
        enemy: EnemyAnalysis,
        terrain: TerrainAnalysis,
    ) -> CollectedIntelligence:
        """Aggregate the three domain analyses into a unified structure."""
        return CollectedIntelligence(
            friendly=friendly,
            enemy=enemy,
            terrain=terrain,
        )
