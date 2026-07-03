"""Checksum and metadata utility functions."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute the checksum of a file.

    Parameters
    ----------
    path:
        Filesystem path to the file.
    algorithm:
        Hash algorithm (default ``"sha256"``).

    Returns
    -------
    str
        Hexadecimal digest string.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_checksum(data: bytes, algorithm: str = "sha256") -> str:
    """Compute the checksum of a byte string.

    Parameters
    ----------
    data:
        Raw byte string.
    algorithm:
        Hash algorithm (default ``"sha256"``).

    Returns
    -------
    str
        Hexadecimal digest string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def compute_json_checksum(data: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Compute a deterministic checksum for a JSON-serialisable structure.

    The JSON is serialised with sorted keys to ensure deterministic
    output across different platforms and Python versions.

    Parameters
    ----------
    data:
        JSON-serialisable object.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    serialised = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return compute_bytes_checksum(serialised)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
