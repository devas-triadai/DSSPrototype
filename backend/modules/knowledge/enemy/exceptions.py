"""Module-specific exceptions for the Enemy Knowledge pipeline."""


class KnowledgeError(RuntimeError):
    """Base exception for all Enemy Knowledge errors."""


class RetrievalError(KnowledgeError):
    """Raised when a knowledge retrieval operation fails."""


class EvidenceError(KnowledgeError):
    """Raised when evidence construction fails."""


class ScoringError(KnowledgeError):
    """Raised when confidence scoring encounters an error."""
