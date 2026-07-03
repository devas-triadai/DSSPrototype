"""Version management for the Ontology Mapping Layer.

Tracks mapping layer version, ontology version compatibility,
and provides version comparison utilities.
"""

from __future__ import annotations

from backend.ontology_mapping.models import MappingVersion


class MappingVersionManager:
    """Manages version tracking for mapping states.

    Version strings use semantic versioning (MAJOR.MINOR.PATCH).
    """

    def __init__(self, initial_version: str = "1.0.0") -> None:
        self._version = initial_version

    @property
    def current_version(self) -> str:
        return self._version

    async def bump_major(self) -> str:
        major, minor, patch = self._parse(self._version)
        self._version = f"{major + 1}.0.0"
        return self._version

    async def bump_minor(self) -> str:
        major, minor, patch = self._parse(self._version)
        self._version = f"{major}.{minor + 1}.0"
        return self._version

    async def bump_patch(self) -> str:
        major, minor, patch = self._parse(self._version)
        self._version = f"{major}.{minor}.{patch + 1}"
        return self._version

    async def compare(
        self,
        a: str,
        b: str,
    ) -> int:
        a_parts = self._parse(a)
        b_parts = self._parse(b)
        for ai, bi in zip(a_parts, b_parts, strict=False):
            if ai < bi:
                return -1
            if ai > bi:
                return 1
        return 0

    async def is_compatible(
        self,
        mapping_version: MappingVersion,
        ontology_version: str,
    ) -> bool:
        """Check if a mapping version is compatible with an ontology version.

        Major version must match for compatibility.
        """
        mapping_major, _, _ = self._parse(mapping_version.ontology_version)
        onto_major, _, _ = self._parse(ontology_version)
        return mapping_major == onto_major

    @staticmethod
    def _parse(version: str) -> tuple[int, int, int]:
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
