"""Schema-level validator — checks document_type and required fields."""

from typing import Any

from knowledge_base.schemas.base import DocumentType
from knowledge_base.validator.base import ValidationResult, Validator

_REQUIRED_FIELDS: dict[DocumentType, set[str]] = {
    DocumentType.FRIENDLY_PLATFORM: {"name", "category"},
    DocumentType.ENEMY_PLATFORM: {"name", "category"},
    DocumentType.TERRAIN_FEATURE: {"terrain_type"},
}


class SchemaValidator(Validator):
    """Validate that each document has a recognised ``document_type``
    and all required fields for that type.
    """

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        for i, doc in enumerate(documents):
            doc_type_raw = doc.get("document_type")
            try:
                doc_type = (
                    DocumentType(doc_type_raw)
                    if isinstance(doc_type_raw, str)
                    else None
                )
            except ValueError:
                errors.append(
                    f"Document [{i}]: unrecognised document_type '{doc_type_raw}'"
                )
                continue

            if doc_type is None:
                errors.append(f"Document [{i}]: missing or invalid document_type")
                continue

            required = _REQUIRED_FIELDS.get(doc_type, set())
            for field_name in required:
                value = doc.get(field_name)
                if not value or (isinstance(value, str) and not value.strip()):
                    errors.append(
                        f"Document [{i}] ({doc_type.value}): missing required "
                        f"field '{field_name}'"
                    )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
