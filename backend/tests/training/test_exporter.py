"""Tests for the ExportPipeline."""

import json
import tempfile
from pathlib import Path

from backend.training.exporter import ExportPipeline


def _make_pipeline() -> tuple[ExportPipeline, Path, Path]:
    tmp = Path(tempfile.mkdtemp())
    return ExportPipeline(exports_dir=tmp / "exports"), tmp, tmp / "models"


def test_export_to_onnx() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    export = pipeline.export_to_onnx("exp_001", "m_001", models_dir)
    assert export.format_name == "onnx"
    assert export.experiment_id == "exp_001"
    assert export.model_id == "m_001"


def test_export_to_torchscript() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    export = pipeline.export_to_torchscript("exp_002", "m_002", models_dir)
    assert export.format_name == "torchscript"


def test_export_to_openvino() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    export = pipeline.export_to_openvino("exp_003", "m_003", models_dir)
    assert export.format_name == "openvino"


def test_list_exports() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    pipeline.export_to_onnx("exp_004", "m_004", models_dir)
    pipeline.export_to_torchscript("exp_004", "m_004", models_dir)
    exports = pipeline.list_exports("m_004")
    assert len(exports) == 2


def test_list_exports_empty() -> None:
    pipeline, _, _ = _make_pipeline()
    assert pipeline.list_exports("nonexistent") == []


def test_export_to_onnx_persists_to_disk() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    pipeline.export_to_onnx("exp_005", "m_005", models_dir)
    file_path = tmp / "exports" / "m_005_onnx.json"
    assert file_path.exists()
    data = json.loads(file_path.read_text())
    assert data["format_name"] == "onnx"


def test_export_to_torchscript_persists_to_disk() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    pipeline.export_to_torchscript("exp_006", "m_006", models_dir)
    file_path = tmp / "exports" / "m_006_torchscript.json"
    assert file_path.exists()


def test_export_to_openvino_persists_to_disk() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    pipeline.export_to_openvino("exp_007", "m_007", models_dir)
    file_path = tmp / "exports" / "m_007_openvino.json"
    assert file_path.exists()


def test_export_onnx_creates_output_dir() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    export = pipeline.export_to_onnx("exp_008", "m_008", models_dir)
    assert "onnx" in export.output_path


def test_export_torchscript_creates_output_dir() -> None:
    pipeline, tmp, models_dir = _make_pipeline()
    export = pipeline.export_to_torchscript("exp_009", "m_009", models_dir)
    assert "torchscript" in export.output_path
    assert export.output_path.endswith(".pt")
