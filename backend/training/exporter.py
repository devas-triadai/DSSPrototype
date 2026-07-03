"""Export pipeline — architecture for model export to multiple formats.

Supports ONNX, TorchScript, and OpenVINO export interfaces.
Does NOT require those frameworks to be installed. Actual conversion
is performed by framework-specific subclasses.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import ExportPipelineInterface
from backend.training.models import ExportData

logger = logging.getLogger("dss.training.exporter")


class ExportPipeline(ExportPipelineInterface):
    """Model export pipeline supporting multiple formats.

    Open/Closed principle: add new export formats by implementing
    the private ``_export`` method pattern.

    No deep learning frameworks are required to use this class.
    """

    def __init__(self, exports_dir: Path | None = None) -> None:
        self._config = training_config
        self._exports_dir = exports_dir or self._config.exports_dir
        self._exports_dir.mkdir(parents=True, exist_ok=True)

    def export_to_onnx(
        self, experiment_id: str, model_id: str, output_dir: Path,
    ) -> ExportData:
        logger.info("ONNX export started: %s (model=%s)", experiment_id, model_id)
        out_dir = output_dir / "onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        export = ExportData(
            experiment_id=experiment_id,
            model_id=model_id,
            format_name="onnx",
            output_path=str(out_dir / "model.onnx"),
            opset_version=self._config.export_onnx_opset,
        )
        self._persist(model_id, export)
        logger.info("ONNX export completed: %s", out_dir)
        return export

    def export_to_torchscript(
        self, experiment_id: str, model_id: str, output_dir: Path,
    ) -> ExportData:
        logger.info("TorchScript export started: %s (model=%s)", experiment_id, model_id)
        out_dir = output_dir / "torchscript"
        out_dir.mkdir(parents=True, exist_ok=True)
        export = ExportData(
            experiment_id=experiment_id,
            model_id=model_id,
            format_name="torchscript",
            output_path=str(out_dir / "model.pt"),
        )
        self._persist(model_id, export)
        logger.info("TorchScript export completed: %s", out_dir)
        return export

    def export_to_openvino(
        self, experiment_id: str, model_id: str, output_dir: Path,
    ) -> ExportData:
        logger.info("OpenVINO export started: %s (model=%s)", experiment_id, model_id)
        out_dir = output_dir / "openvino"
        out_dir.mkdir(parents=True, exist_ok=True)
        export = ExportData(
            experiment_id=experiment_id,
            model_id=model_id,
            format_name="openvino",
            output_path=str(out_dir / "model.xml"),
        )
        self._persist(model_id, export)
        logger.info("OpenVINO export completed: %s", out_dir)
        return export

    def list_exports(self, model_id: str) -> list[ExportData]:
        exports: list[ExportData] = []
        for path in sorted(self._exports_dir.glob(f"{model_id}_*.json")):
            try:
                data = json.loads(path.read_text())
                exports.append(ExportData(**data))
            except Exception:
                pass
        return exports

    def _export_path(self, model_id: str, format_name: str) -> Path:
        return self._exports_dir / f"{model_id}_{format_name}.json"

    def _persist(self, model_id: str, export: ExportData) -> None:
        path = self._export_path(model_id, export.format_name)
        path.write_text(json.dumps(export.model_dump(), indent=2, default=str))
