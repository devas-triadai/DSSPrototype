"""Dataset path definitions for the DSS Knowledge Base.

All built-in dataset paths are defined here so that the registry
and retrievers can reference them by import.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

FRIENDLY_PATH = str(_HERE / "friendly_platforms.json")
ENEMY_PATH = str(_HERE / "enemy_platforms.json")
TERRAIN_PATH = str(_HERE / "terrain_features.json")

__all__ = [
    "FRIENDLY_PATH",
    "ENEMY_PATH",
    "TERRAIN_PATH",
]
