"""Command-line interface for the Training Platform.

Usage:
    python -m backend.training.cli train --help
    python -m backend.training train --dataset coco2017 --data /data.yaml --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from backend.training.backends.yolo_backend import YOLOTrainingBackend
from backend.training.checkpoint import CheckpointManager
from backend.training.dataset_exporter import SUPPORTED_DATASET_TYPES, TrainingDatasetExporter
from backend.training.exceptions import DatasetNotReadyError, PreValidationError
from backend.training.experiment import ExperimentManager
from backend.training.exporter import ExportPipeline
from backend.training.history import HistoryManager
from backend.training.metrics import MetricsManager
from backend.training.models import TrainingConfigData
from backend.training.service import TrainingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("dss.training.cli")

ARTIFACTS_DIR = Path("artifacts") / "training"


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dss-train",
        description="DSS Training Platform CLI — exposes the existing "
        "TrainingService and TrainingPipeline for YOLO model training.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    _add_train_parser(sub)
    _add_export_parser(sub)
    return parser


def _add_train_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train", help="Run a training job")

    p.add_argument("--dataset", required=True, help="Dataset name (e.g. coco2017)")
    p.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to YOLO dataset YAML file or directory containing data.yaml",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. yolo11n.pt) or path to a .pt checkpoint",
    )

    p.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="Batch size (default: 16)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    p.add_argument("--workers", type=int, default=4, help="Data loader workers (default: 4)")
    p.add_argument("--device", default="cpu", help='Device: "cpu" or "cuda:0" (default: cpu)')
    p.add_argument(
        "--project",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Project output directory (default: artifacts/training)",
    )
    p.add_argument("--name", default=None, help="Experiment name (default: same as --dataset)")
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: no early stopping)",
    )
    p.add_argument("--optimizer", default="Adam", help='Optimizer (default: "Adam")')
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        dest="weight_decay",
        help="Weight decay (default: 0.0001)",
    )
    p.add_argument(
        "--save-period",
        type=int,
        default=5,
        dest="save_period",
        help="Checkpoint save interval in epochs (default: 5)",
    )
    p.add_argument(
        "--export",
        action="store_true",
        help="Export model to ONNX and TorchScript after training",
    )


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []

    if not args.data.exists():
        errors.append(f"Data path not found: {args.data}")
    else:
        _validate_data_path(args.data, errors)

    if args.batch_size < 1:
        errors.append("--batch-size must be >= 1")

    if args.epochs < 1:
        errors.append("--epochs must be >= 1")

    if args.imgsz < 32:
        errors.append("--imgsz must be >= 32")

    if args.lr <= 0:
        errors.append("--lr must be positive")

    if args.weight_decay < 0:
        errors.append("--weight-decay must be non-negative")

    if args.save_period < 1:
        errors.append("--save-period must be >= 1")

    _validate_device(args.device, errors)

    if errors:
        for e in errors:
            LOG.error("%s", e)
        sys.exit(1)


def _validate_data_path(data_path: Path, errors: list[str]) -> None:
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
        if not data_yaml.exists():
            errors.append(
                f"No data.yaml found in {data_path}. "
                f"Please provide a YOLO dataset YAML file via --data.",
            )


def _validate_device(device: str, errors: list[str]) -> None:
    if device == "cpu":
        return
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append(f"CUDA device '{device}' requested but CUDA is not available")
        except ImportError:
            errors.append(
                f"CUDA device '{device}' requested but PyTorch is not installed",
            )
    else:
        errors.append(f"Invalid device: '{device}'. Use 'cpu' or 'cuda:N'.")


# ------------------------------------------------------------------
# Data YAML resolution
# ------------------------------------------------------------------


def _resolve_data_yaml(experiment_name: str, data_path: Path) -> Path:
    """Copy the user's data YAML to the location the Trainer expects.

    The Trainer constructs the path ``datasets/exports/yolo/{dataset_version}/data.yaml``
    in its ``_resolve_dataset_path`` method.  This function ensures that file
    exists so the YOLO backend receives a valid data configuration.
    """
    target_dir = Path("datasets") / "exports" / "yolo" / experiment_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "data.yaml"

    if data_path.is_file():
        shutil.copy2(data_path, target)
    else:
        inner = data_path / "data.yaml"
        if inner.exists():
            shutil.copy2(inner, target)
        else:
            raise FileNotFoundError(f"No data.yaml found at {data_path}")

    return target


# ------------------------------------------------------------------
# Configuration builder
# ------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> TrainingConfigData:
    return TrainingConfigData(
        model_name=args.model,
        dataset_version=args.name or args.dataset,
        experiment_name=args.name or args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        optimizer=args.optimizer.lower(),
        weight_decay=args.weight_decay,
        image_size=(args.imgsz, args.imgsz),
        device=str(args.device),
        workers=args.workers,
        seed=args.seed,
        resume_checkpoint=args.resume,
        early_stopping_patience=args.patience,
        save_interval=args.save_period,
    )


# ------------------------------------------------------------------
# Export subcommand
# ------------------------------------------------------------------


def _add_export_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("export", help="Export a raw dataset to YOLO-ready format")

    supported = ", ".join(sorted(SUPPORTED_DATASET_TYPES))
    p.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset type ({supported})",
    )
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the raw dataset directory",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Export directory name (default: source directory basename)",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=Path("datasets") / "exports" / "yolo",
        help="Base output directory (default: datasets/exports/yolo)",
    )


def cmd_export(args: argparse.Namespace) -> None:
    exporter = TrainingDatasetExporter(output_base=args.output_base)
    source = args.source.resolve()

    LOG.info("Exporting dataset '%s' from %s ...", args.dataset, source)

    result = exporter.export(
        dataset_type=args.dataset,
        source_path=source,
        dataset_name=args.name,
    )

    if result["status"] == "failed":
        LOG.error("Export failed:")
        for err in result["errors"]:
            LOG.error("  - %s", err)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Dataset export completed")
    print(f"  Name       : {result['dataset_name']}")
    print(f"  Type       : {result['dataset_type']}")
    print(f"  Classes    : {len(result['class_names'])}")
    print(f"  Train      : {result['train_count']} images")
    print(f"  Val        : {result['val_count']} images")
    print(f"  Annotations: {result['annotation_count']}")
    print(f"  Output     : {result['output_dir']}")
    print(f"  Data YAML  : {result['data_yaml']}")
    print(f"{'=' * 60}\n")

    LOG.info("Export completed: %s", result["output_dir"])


# ------------------------------------------------------------------
# Service factory
# ------------------------------------------------------------------


def _build_service(experiment_name: str) -> TrainingService:
    artifacts = ARTIFACTS_DIR / experiment_name
    return TrainingService(
        checkpoint_manager=CheckpointManager(
            checkpoints_dir=artifacts / "weights",
        ),
        experiment_manager=ExperimentManager(experiments_dir=artifacts),
        history_manager=HistoryManager(history_dir=artifacts),
        metrics_manager=MetricsManager(metrics_dir=artifacts),
        export_pipeline=ExportPipeline(exports_dir=artifacts / "exports"),
    )


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    experiment_name = args.name or args.dataset

    _resolve_data_yaml(experiment_name, args.data)

    config = _build_config(args)

    service = _build_service(experiment_name)
    service.set_training_backend(YOLOTrainingBackend())

    _print_header(experiment_name, args)

    try:
        result = service.train(config)
    except (DatasetNotReadyError, PreValidationError) as exc:
        LOG.error("Pre-training validation failed: %s", exc)
        sys.exit(1)

    _print_results(result)

    if args.export:
        _run_export(service, result, experiment_name)


def _run_export(
    service: TrainingService,
    result: object,
    experiment_name: str,
) -> None:
    from backend.training.models import TrainingResult

    if not isinstance(result, TrainingResult):
        LOG.warning("Cannot export: unexpected result type %s", type(result).__name__)
        return

    fmt_list = ["onnx", "torchscript"]
    LOG.info("Exporting model to %s ...", ", ".join(fmt_list))

    try:
        exports = service.export_model(
            experiment_id=result.experiment_id,
            model_id=result.model_id,
            output_dir=ARTIFACTS_DIR / experiment_name / "exports",
            formats=fmt_list,
        )
    except Exception as exc:
        LOG.error("Export failed: %s", exc)
        return

    for exp_data in exports:
        LOG.info("  Exported: %s -> %s", exp_data.format_name, exp_data.output_path)


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------


def _print_header(experiment_name: str, args: argparse.Namespace) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DSS Training Platform")
    print(f"  Experiment : {experiment_name}")
    print(f"  Model      : {args.model}")
    print(f"  Data       : {args.data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch_size}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"{sep}\n")


def _print_results(result: object) -> None:
    from backend.training.models import TrainingResult

    if not isinstance(result, TrainingResult):
        LOG.warning("Unexpected result type: %s", type(result).__name__)
        return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Training completed: {result.status}")
    print(f"  Experiment ID     : {result.experiment_id}")
    print(f"  Model ID          : {result.model_id}")
    print(f"  Epochs completed  : {result.total_epochs_completed}")
    print(f"  Best epoch        : {result.best_epoch}")
    print(f"  Best metric       : {result.best_metric_name} = {result.best_metric}")
    print(f"  Duration          : {result.training_duration_seconds:.1f}s")
    if result.final_metrics:
        fm = result.final_metrics
        if fm.mAP50 is not None:
            print(f"  mAP50             : {fm.mAP50:.4f}")
        if fm.mAP50_95 is not None:
            print(f"  mAP50-95          : {fm.mAP50_95:.4f}")
    print(f"{sep}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        validate_args(args)
        cmd_train(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
