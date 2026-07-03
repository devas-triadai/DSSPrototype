"""Tests for the ChecksumGenerator."""

import tempfile
from pathlib import Path

from backend.dataset_manager.checksum import ChecksumGenerator
from backend.dataset_manager.models import DatasetChecksum


def _write_file(path: Path, content: str = "hello world") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_compute_file_checksum() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        file = _write_file(Path(tmp) / "test.txt")
        cs = ChecksumGenerator().compute(file)
        assert isinstance(cs, DatasetChecksum)
        assert cs.algorithm == "sha256"
        assert len(cs.value) == 64  # SHA256 hex digest
        assert cs.file_path == str(file.resolve())


def test_verify_matching() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        file = _write_file(Path(tmp) / "verify.txt")
        cs = ChecksumGenerator().compute(file)
        assert ChecksumGenerator().verify(file, cs.value) is True


def test_verify_not_matching() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        file = _write_file(Path(tmp) / "bad.txt")
        ChecksumGenerator().compute(file)
        bad_hash = "0" * 64
        assert ChecksumGenerator().verify(file, bad_hash) is False


def test_compute_directory_checksum() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_file(d / "a.txt", "aaa")
        _write_file(d / "b.txt", "bbb")
        cs = ChecksumGenerator().compute(d)
        assert len(cs.value) == 64


def test_deterministic_directory_checksum() -> None:
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        for d in [Path(tmp1), Path(tmp2)]:
            _write_file(d / "f1.txt", "content1")
            _write_file(d / "f2.txt", "content2")

        cs1 = ChecksumGenerator().compute(Path(tmp1))
        cs2 = ChecksumGenerator().compute(Path(tmp2))
        assert cs1.value == cs2.value
