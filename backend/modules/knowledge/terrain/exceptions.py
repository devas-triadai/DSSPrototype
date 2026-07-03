"""Module-specific exceptions for the Terrain Knowledge pipeline."""


class KnowledgeError(RuntimeError):
    """Base exception for all Terrain Knowledge errors."""


class RetrievalError(KnowledgeError):
    """Raised when a terrain retrieval operation fails."""


class TerrainError(KnowledgeError):
    """Raised when terrain feature extraction fails."""


class MobilityError(KnowledgeError):
    """Raised when mobility analysis encounters an error."""


class VisibilityError(KnowledgeError):
    """Raised when visibility analysis encounters an error."""


class ScoringError(KnowledgeError):
    """Raised when confidence scoring encounters an error."""
