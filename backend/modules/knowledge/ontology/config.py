"""Ontology configuration.

Paths and defaults for the ontology layer.
"""

from pathlib import Path
from typing import Final

# Base ontology directory — relative to project root.
ONTOLOGY_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "knowledge_base"
    / "ontology"
)

# Default ontology files by domain.
DEFAULT_ONTOLOGY_FILES: Final[dict[str, Path]] = {
    "vehicles": ONTOLOGY_DIR / "vehicles.json",
    "aircraft": ONTOLOGY_DIR / "aircraft.json",
    "drones": ONTOLOGY_DIR / "drones.json",
    "weapons": ONTOLOGY_DIR / "weapons.json",
    "terrain": ONTOLOGY_DIR / "terrain.json",
}

# Confidence decay per hierarchy level (multiplicative).
CATEGORY_CONFIDENCE_DECAY: Final[float] = 0.90

# Confidence decay for alias mappings (multiplicative).
ALIAS_CONFIDENCE_DECAY: Final[float] = 0.92

# Confidence decay for military-equivalent mappings.
MILITARY_EQUIVALENT_DECAY: Final[float] = 0.88

# Confidence decay for civilian-equivalent mappings.
CIVILIAN_EQUIVALENT_DECAY: Final[float] = 0.88

# Maximum number of parent levels to traverse.
MAX_PARENT_DEPTH: Final[int] = 4

# Maximum number of expanded concepts to return.
MAX_EXPANDED_CONCEPTS: Final[int] = 12
