"""Version consistency validator — ensures all documents use the same
document version for a given type."""

from typing import Any

from knowledge_base.validator.base import ValidationResult, Validator


class VersionValidator(Validator):
    """Check that all documents of the same ``document_type`` use the
    same ``version`` string (from ``metadata``).

    Inconsistent versions produce warnings but do not fail validation.
    """

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        versions_by_type: dict[str, set[str]] = {}

        for i, doc in enumerate(documents):
            doc_type = doc.get("document_type", "unknown")
            metadata = doc.get("metadata", {})
            version = metadata.get("version", "1.0.0") if isinstance(metadata, dict) else "1.0.0"
            versions_by_type.setdefault(doc_type, set()).add(version)

        for doc_type, versions in versions_by_type.items():
            if len(versions) > 1:
                warnings.append(
                    f"Document type '{doc_type}' has mixed versions: "
                    f"{', '.join(sorted(versions))}"
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
