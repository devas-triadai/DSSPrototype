"""Abstract validator interface and shared result type."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a set of documents.

    Attributes
    ----------
    valid:
        ``True`` if all documents passed validation.
    errors:
        Per-document validation error messages.
    warnings:
        Non-fatal warnings.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Validator(Protocol):
    """Interface for all document validators."""

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult: ...  # pragma: no cover
