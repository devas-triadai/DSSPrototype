"""Tests for the EvaluationEngine."""

import json
import tempfile
from pathlib import Path

from backend.training.evaluator import EvaluationEngine


def _make_engine() -> tuple[EvaluationEngine, Path]:
    tmp = Path(tempfile.mkdtemp())
    return EvaluationEngine(reports_dir=tmp / "reports"), tmp


def test_validate() -> None:
    engine, _ = _make_engine()
    result = engine.validate("exp_001", "/tmp/ckpt.pt", "v1.0.0")
    assert result.experiment_id == "exp_001"
    assert result.split == "validation"
    assert result.dataset_version == "v1.0.0"


def test_test() -> None:
    engine, _ = _make_engine()
    result = engine.test("exp_002", "/tmp/ckpt.pt")
    assert result.split == "test"


def test_benchmark() -> None:
    engine, _ = _make_engine()
    result = engine.benchmark("exp_003", "/tmp/ckpt.pt")
    assert result.split == "benchmark"


def test_evaluation_result_defaults() -> None:
    engine, _ = _make_engine()
    result = engine.validate("exp_004", "/tmp/ckpt.pt")
    assert result.mAP50 is None
    assert result.precision is None
    assert result.recall is None


def test_validate_persists_to_disk() -> None:
    engine, tmp = _make_engine()
    engine.validate("exp_005", "/tmp/ckpt.pt", "v1.0")
    file_path = tmp / "reports" / "exp_005_validation_eval.json"
    assert file_path.exists()
    data = json.loads(file_path.read_text())
    assert data["experiment_id"] == "exp_005"


def test_test_persists_to_disk() -> None:
    engine, tmp = _make_engine()
    engine.test("exp_006", "/tmp/ckpt.pt")
    file_path = tmp / "reports" / "exp_006_test_eval.json"
    assert file_path.exists()


def test_benchmark_persists_to_disk() -> None:
    engine, tmp = _make_engine()
    engine.benchmark("exp_007", "/tmp/ckpt.pt")
    file_path = tmp / "reports" / "exp_007_benchmark_eval.json"
    assert file_path.exists()


def test_validate_returns_correct_split() -> None:
    engine, _ = _make_engine()
    result = engine.validate("exp_008", "/tmp/ckpt.pt")
    assert result.split == "validation"


def test_test_returns_correct_split() -> None:
    engine, _ = _make_engine()
    result = engine.test("exp_009", "/tmp/ckpt.pt")
    assert result.split == "test"


def test_benchmark_returns_correct_split() -> None:
    engine, _ = _make_engine()
    result = engine.benchmark("exp_010", "/tmp/ckpt.pt")
    assert result.split == "benchmark"
