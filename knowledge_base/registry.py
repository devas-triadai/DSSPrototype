"""Dataset registry — maps dataset names to retriever classes.

Provides a central registry of all datasets available in the
knowledge base.  Used by external modules to discover and
instantiate retrievers without hardcoding import paths.
"""

from knowledge_base.datasets import ENEMY_PATH, FRIENDLY_PATH, TERRAIN_PATH
from knowledge_base.enemy.retriever import EnemyKnowledgeRetriever
from knowledge_base.friendly.retriever import FriendlyKnowledgeRetriever
from knowledge_base.terrain.retriever import TerrainKnowledgeRetriever


class DatasetRegistryEntry:
    """Metadata for a registered dataset."""

    def __init__(
        self,
        name: str,
        description: str,
        retriever_class: type,
        default_dataset_path: str,
    ) -> None:
        self.name = name
        self.description = description
        self.retriever_class = retriever_class
        self.default_dataset_path = default_dataset_path


_REGISTRY: dict[str, DatasetRegistryEntry] = {}


def register_dataset(
    name: str,
    description: str,
    retriever_class: type,
    default_dataset_path: str,
) -> None:
    """Register a dataset in the global registry."""
    _REGISTRY[name] = DatasetRegistryEntry(
        name=name,
        description=description,
        retriever_class=retriever_class,
        default_dataset_path=default_dataset_path,
    )


def get_dataset_names() -> list[str]:
    """Return the names of all registered datasets."""
    return list(_REGISTRY.keys())


def get_dataset_registry_entry(name: str) -> DatasetRegistryEntry | None:
    """Look up a dataset by name."""
    return _REGISTRY.get(name)


def list_datasets() -> list[dict[str, str]]:
    """Return summary info for all registered datasets."""
    return [
        {
            "name": entry.name,
            "description": entry.description,
        }
        for entry in _REGISTRY.values()
    ]


# --- Register built-in datasets ---

register_dataset(
    name="friendly_platforms",
    description="Public Indian military equipment data",
    retriever_class=FriendlyKnowledgeRetriever,
    default_dataset_path=FRIENDLY_PATH,
)

register_dataset(
    name="enemy_platforms",
    description="Public foreign military equipment data (potential adversaries)",
    retriever_class=EnemyKnowledgeRetriever,
    default_dataset_path=ENEMY_PATH,
)

register_dataset(
    name="terrain_features",
    description="Public terrain features for Northern India border region",
    retriever_class=TerrainKnowledgeRetriever,
    default_dataset_path=TERRAIN_PATH,
)
