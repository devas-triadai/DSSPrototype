"""Canonical knowledge document schemas for the DSS Knowledge Base.

Each schema defines the required and optional fields for a type of
knowledge document.  Documents MUST conform to these schemas to be
accepted by the validator and indexer.

Current schemas:

* ``FriendlyPlatformDocument`` — friendly military equipment and units
* ``EnemyPlatformDocument`` — enemy military equipment and units
* ``TerrainFeatureDocument`` — terrain features and geographic data
"""

from knowledge_base.schemas.base import BaseDocument, DocumentType
from knowledge_base.schemas.enemy_platform import EnemyPlatformDocument
from knowledge_base.schemas.friendly_platform import FriendlyPlatformDocument
from knowledge_base.schemas.terrain_feature import TerrainFeatureDocument

__all__ = [
    "BaseDocument",
    "DocumentType",
    "FriendlyPlatformDocument",
    "EnemyPlatformDocument",
    "TerrainFeatureDocument",
]
