"""Abstract interfaces (contracts) for every DSS AI module.

Future module implementations **must** inherit from these interfaces
to guarantee the system can wire them together at runtime.
"""

from backend.contracts.interfaces.decision import DecisionModule
from backend.contracts.interfaces.enemy import EnemyModule
from backend.contracts.interfaces.friendly import FriendlyModule
from backend.contracts.interfaces.fusion import FusionModule
from backend.contracts.interfaces.terrain import TerrainModule
from backend.contracts.interfaces.vision import VisionModule

__all__ = [
    "VisionModule",
    "FriendlyModule",
    "EnemyModule",
    "TerrainModule",
    "FusionModule",
    "DecisionModule",
]
