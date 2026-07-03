"""Reusable exception hierarchy for the DSS prototype.

All custom exceptions inherit from ``DSSBaseError`` so that
top-level handlers can catch a single base type when needed.
"""


class DSSBaseError(Exception):
    """Base exception for all DSS-prototype errors."""

    def __init__(self, message: str, code: str = "UNKNOWN") -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)


class ConfigurationError(DSSBaseError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


class ModuleError(DSSBaseError):
    """Raised when an AI module encounters a runtime error."""

    def __init__(self, message: str, module_name: str = "") -> None:
        detail = f"[{module_name}] {message}" if module_name else message
        super().__init__(detail, code="MODULE_ERROR")


class ServiceError(DSSBaseError):
    """Raised when a service-layer operation fails."""

    def __init__(self, message: str, service_name: str = "") -> None:
        detail = f"[{service_name}] {message}" if service_name else message
        super().__init__(detail, code="SERVICE_ERROR")


class DatabaseError(DSSBaseError):
    """Raised on database connection or query failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="DATABASE_ERROR")


class FileOperationError(DSSBaseError):
    """Raised when a file read / write / upload operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="FILE_ERROR")
