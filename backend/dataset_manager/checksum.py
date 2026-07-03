"""Checksum generator — SHA256 integrity verification for datasets.

Supports checksum computation for files and directories, integrity
verification, and persistence of checksum manifests.
"""

import hashlib
import json
import logging
from pathlib import Path

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import ChecksumInterface
from backend.dataset_manager.models import DatasetChecksum

logger = logging.getLogger("dss.dataset_manager.checksum")


class ChecksumGenerator(ChecksumInterface):
    """SHA256-based checksum generation and verification."""

    def __init__(self) -> None:
        self._config = dm_config
        self._algorithm = self._config.checksum_algorithm

    def compute(self, path: Path) -> DatasetChecksum:
        logger.info("Checksum computation started: %s", path)

        if path.is_file():
            value = self._hash_file(path)
        elif path.is_dir():
            value = self._hash_directory(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        checksum = DatasetChecksum(
            algorithm=self._algorithm,
            value=value,
            file_path=str(path.resolve()),
        )

        self._persist(checksum, path)
        logger.info("Checksum computed: %s", value[:16])
        return checksum

    def verify(self, path: Path, expected: str) -> bool:
        logger.info("Checksum verification started: %s", path)
        actual = self.compute(path)
        match = actual.value == expected
        if match:
            logger.info("Checksum verified: %s", path)
        else:
            logger.warning("Checksum mismatch: %s", path)
        return match

    def verify_against_file(self, path: Path, checksum_file: Path) -> bool:
        """Verify *path* against a previously persisted checksum manifest."""
        if not checksum_file.exists():
            logger.error("Checksum file not found: %s", checksum_file)
            return False
        try:
            data = json.loads(checksum_file.read_text())
            expected = data.get("value", "")
            return self.verify(path, expected)
        except Exception as e:
            logger.error("Failed to read checksum file: %s", e)
            return False

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _hash_directory(self, path: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(path.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(path).as_posix().encode())
                h.update(self._hash_file(p).encode())
        return h.hexdigest()

    def _persist(self, checksum: DatasetChecksum, source: Path) -> None:
        checksum_dir = self._config.checksums_dir
        checksum_dir.mkdir(parents=True, exist_ok=True)
        dest = checksum_dir / f"{source.name}_checksum.json"
        dest.write_text(json.dumps(checksum.model_dump(), indent=2))
