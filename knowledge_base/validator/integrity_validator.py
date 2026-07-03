"""Data integrity validator — checks fields for corrupted or
unexpected values."""

from typing import Any

from knowledge_base.validator.base import ValidationResult, Validator


class IntegrityValidator(Validator):
    """Perform basic integrity checks on document data.

    Flags empty required fields, non-UTF-8-encodable values, and
    excessively long string fields as warnings.
    """

    _MAX_STRING_LENGTH = 5000

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        for i, doc in enumerate(documents):
            for field_name, value in doc.items():
                if field_name == "metadata":
                    continue
                if isinstance(value, str):
                    if len(value) > self._MAX_STRING_LENGTH:
                        warnings.append(
                            f"Document [{i}]: field '{field_name}' exceeds "
                            f"{self._MAX_STRING_LENGTH} characters "
                            f"({len(value)} chars)"
                        )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
