"""Field-level validator — checks value types, ranges, and formats."""

from typing import Any

from knowledge_base.validator.base import ValidationResult, Validator

_STRING_FIELDS = {
    "name", "category", "platform_type", "country", "manufacturer",
    "role", "terrain_type", "region", "soil_type", "visibility",
    "tactical_role",
}

_FLOAT_FIELDS = {
    "weight_tonnes", "speed_kmh", "elevation", "slope",
}

_INT_FIELDS = {"crew"}

_LIST_FIELDS = {
    "characteristics", "equipment", "markings", "armament", "sensors",
    "capabilities", "threat_indicators", "countermeasures", "features",
    "road_network", "water_bodies", "vegetation", "obstacles",
}


class FieldValidator(Validator):
    """Validate that document field values have the correct types.

    Checks string, float, int, and list fields.  Non-fatal warnings
    are emitted for type mismatches rather than hard errors.
    """

    def validate(
        self,
        documents: list[dict[str, Any]],
    ) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        for i, doc in enumerate(documents):
            for field_name, value in doc.items():
                if field_name in ("document_type", "metadata"):
                    continue
                if field_name in _STRING_FIELDS:
                    if value is not None and not isinstance(value, str):
                        warnings.append(
                            f"Document [{i}]: field '{field_name}' should be str, "
                            f"got {type(value).__name__}"
                        )
                elif field_name in _FLOAT_FIELDS:
                    if value is not None and not isinstance(value, (int, float)):
                        warnings.append(
                            f"Document [{i}]: field '{field_name}' should be numeric, "
                            f"got {type(value).__name__}"
                        )
                elif field_name in _INT_FIELDS:
                    if value is not None and not isinstance(value, int):
                        warnings.append(
                            f"Document [{i}]: field '{field_name}' should be int, "
                            f"got {type(value).__name__}"
                        )
                elif field_name in _LIST_FIELDS:
                    if value is not None and not isinstance(value, list):
                        warnings.append(
                            f"Document [{i}]: field '{field_name}' should be list, "
                            f"got {type(value).__name__}"
                        )

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
