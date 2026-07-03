"""Duplicate document detection based on document_id or (name, category)."""

from typing import Any

from knowledge_base.validator.base import ValidationResult, Validator


class DuplicateValidator(Validator):
    """Detect documents with duplicate ``document_id`` or duplicate
    ``(name, category)`` pairs within the same dataset load.

    The first occurrence is kept; subsequent occurrences are flagged
    as errors.
    """

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        seen_name_cat: set[tuple[str, str]] = set()

        for i, doc in enumerate(documents):
            doc_id = doc.get("document_id") or doc.get("metadata", {}).get("document_id")
            if doc_id and isinstance(doc_id, str):
                if doc_id in seen_ids:
                    errors.append(
                        f"Document [{i}]: duplicate document_id '{doc_id}'"
                    )
                else:
                    seen_ids.add(doc_id)

            name = doc.get("name", "")
            category = doc.get("category", "")
            if name and category:
                key = (name, category)
                if key in seen_name_cat:
                    warnings.append(
                        f"Document [{i}]: duplicate (name='{name}', "
                        f"category='{category}')"
                    )
                else:
                    seen_name_cat.add(key)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
