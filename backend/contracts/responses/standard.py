"""Reusable response envelopes for every DSS API endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SuccessResponse(BaseModel):
    """Envelope returned on a successful API operation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True, description="Indicates the operation succeeded")
    message: str = Field(..., min_length=1, description="Human-readable status message")
    data: dict[str, Any] | None = Field(
        None, description="Optional payload returned with the response"
    )


class ErrorResponse(BaseModel):
    """Envelope returned when an API operation fails."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=False, description="Indicates the operation failed")
    error_code: str = Field(..., min_length=1, description="Machine-readable error code")
    message: str = Field(..., min_length=1, description="Human-readable error description")
    details: dict[str, Any] | None = Field(
        None, description="Optional structured error details"
    )


class ValidationResponse(BaseModel):
    """Envelope returned for request-validation failures."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(
        default_factory=list, description="List of validation error messages"
    )
