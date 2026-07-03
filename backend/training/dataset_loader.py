"""Dataset loader — bridges Dataset Quality to Training.

Only production-approved datasets may pass through.
Rejects datasets that are not production_ready.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.exceptions import DatasetNotReadyError
from backend.training.interfaces import DatasetLoaderInterface
from backend.training.models import DatasetLoadResult

logger = logging.getLogger("dss.training.dataset_loader")


class DatasetLoader(DatasetLoaderInterface):
    """Loads datasets that have passed Dataset Quality checks.

    Scans a quality reports directory for JSON reports from the
    Dataset Quality pipeline. Only datasets with production_ready=True
    are made available for training.
    """

    def __init__(self, reports_dir: Path | None = None) -> None:
        self._config = training_config
        self._reports_dir = reports_dir or (self._config.base_dir / "dataset_quality" / "reports")
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(
        self,
        dataset_name: str,
        dataset_version: str = "",
    ) -> DatasetLoadResult:
        logger.info("Loading dataset: %s v%s", dataset_name, dataset_version or "latest")

        report = self._find_report(dataset_name, dataset_version)
        if report is None:
            raise DatasetNotReadyError(
                f"No quality report found for dataset '{dataset_name}' "
                f"(version: {dataset_version or 'latest'})",
            )

        if not report.get("production_ready", False):
            raise DatasetNotReadyError(
                f"Dataset '{dataset_name}' is not production ready. "
                f"Run Dataset Quality pipeline first.",
            )

        result = self._report_to_result(report, dataset_name, dataset_version)
        logger.info(
            "Dataset loaded: %s v%s (%d images, %d classes)",
            result.dataset_name, result.dataset_version,
            result.total_images, result.num_classes,
        )
        return result

    def list_available_datasets(self) -> list[DatasetLoadResult]:
        results: list[DatasetLoadResult] = []
        for path in sorted(self._reports_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                if data.get("overall_score", {}).get("production_ready", False):
                    name = data.get("dataset_name", path.stem)
                    version = data.get("dataset_version", "")
                    results.append(self._report_to_result(data, name, version))
            except Exception:
                pass
        return results

    def _find_report(
        self, dataset_name: str, dataset_version: str,
    ) -> dict[str, object] | None:
        candidates = list(self._reports_dir.glob(f"{dataset_name}_*.json"))
        if dataset_version:
            candidates = [p for p in candidates if dataset_version in p.name]
        if not candidates:
            candidates = list(self._reports_dir.glob(f"{dataset_name}.json"))
        if not candidates:
            return None
        try:
            data = json.loads(candidates[0].read_text())
            assert isinstance(data, dict)
            return data
        except Exception:
            return None

    def _report_to_result(
        self, report: dict[str, object], dataset_name: str, dataset_version: str,
    ) -> DatasetLoadResult:
        score = report.get("overall_score", {})
        if isinstance(score, dict):
            production_ready = bool(score.get("production_ready", False))
        else:
            production_ready = bool(getattr(score, "production_ready", False))

        class_names = report.get("class_names", [])
        num_classes_val = report.get("num_classes", 0)
        total_images_val = report.get("total_images", 0)
        total_annotations_val = report.get("total_annotations", 0)
        return DatasetLoadResult(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            quality_report_version=str(report.get("pipeline_version", "")),
            production_ready=production_ready,
            train_path=str(Path("datasets") / "exports" / "yolo" / dataset_version / "train"),
            val_path=str(Path("datasets") / "exports" / "yolo" / dataset_version / "val"),
            test_path=str(Path("datasets") / "exports" / "yolo" / dataset_version / "test"),
            class_names=list(class_names) if isinstance(class_names, (list, tuple)) else [],
            num_classes=(
                int(num_classes_val) if isinstance(num_classes_val, (int, float)) else 0
            ),
            total_images=(
                int(total_images_val) if isinstance(total_images_val, (int, float)) else 0
            ),
            total_annotations=(
                int(total_annotations_val) if isinstance(total_annotations_val, (int, float)) else 0
            ),
        )
