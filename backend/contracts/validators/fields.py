"""Field-level validators for DSS domain models."""

from backend.contracts.models.geometry import BoundingBox


def validate_bounding_box(bbox: BoundingBox) -> BoundingBox:
    """Ensure a bounding box has strictly positive dimensions.

    Parameters
    ----------
    bbox:
        The bounding box to validate.

    Returns
    -------
    BoundingBox
        The same bounding box if valid.

    Raises
    ------
    ValueError
        If width or height is zero or negative.
    """
    if bbox.width <= 0:
        raise ValueError("BoundingBox width must be positive")
    if bbox.height <= 0:
        raise ValueError("BoundingBox height must be positive")
    return bbox


def validate_confidence(value: float) -> float:
    """Ensure a numeric confidence value lies in the range [0, 1].

    Parameters
    ----------
    value:
        The confidence value to check.

    Returns
    -------
    float
        The same value if valid.

    Raises
    ------
    ValueError
        If the value is outside [0, 1].
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {value}")
    return value
