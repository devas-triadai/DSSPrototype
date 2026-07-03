"""Core infrastructure package.

Provides foundational components used by all other modules:
  - setup_logging: Centralized logging initialization
  - BaseModule: Abstract base class for all AI modules
  - Custom exception hierarchy for structured error handling
"""

from backend.core.base import BaseModule
from backend.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    DSSBaseError,
    FileOperationError,
    ModuleError,
    ServiceError,
)
from backend.core.logging import setup_logging

__all__ = [
    "setup_logging",
    "DSSBaseError",
    "ConfigurationError",
    "ModuleError",
    "ServiceError",
    "DatabaseError",
    "FileOperationError",
    "BaseModule",
]
