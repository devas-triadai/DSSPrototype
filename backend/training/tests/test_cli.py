"""Tests for the Training CLI.

Covers argument parsing, validation, configuration generation,
data YAML resolution, service factory, and output helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.training.cli import (
    _build_config,
    _build_service,
    _print_header,
    _print_results,
    _resolve_data_yaml,
    build_parser,
    main,
    validate_args,
)
from backend.training.models import TrainingConfigData, TrainingResult

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------


class TestArgParsing:
    def test_minimal_args(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "coco2017", "--data", "data.yaml", "--model", "yolo11n.pt",
        ])
        assert args.command == "train"
        assert args.dataset == "coco2017"
        assert args.data == Path("data.yaml")
        assert args.model == "yolo11n.pt"

    def test_all_options(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train",
            "--dataset", "voc2012",
            "--data", "/path/to/data.yaml",
            "--model", "yolo11x.pt",
            "--epochs", "200",
            "--batch-size", "32",
            "--imgsz", "1280",
            "--workers", "8",
            "--device", "cuda:0",
            "--project", "/outputs",
            "--name", "voc_experiment",
            "--resume", "checkpoint.pt",
            "--seed", "7",
            "--patience", "20",
            "--optimizer", "SGD",
            "--lr", "0.01",
            "--weight-decay", "0.0005",
            "--save-period", "10",
            "--export",
        ])
        assert args.dataset == "voc2012"
        assert args.epochs == 200
        assert args.batch_size == 32
        assert args.imgsz == 1280
        assert args.workers == 8
        assert args.device == "cuda:0"
        assert args.project == Path("/outputs")
        assert args.name == "voc_experiment"
        assert args.resume == "checkpoint.pt"
        assert args.seed == 7
        assert args.patience == 20
        assert args.optimizer == "SGD"
        assert args.lr == 0.01
        assert args.weight_decay == 0.0005
        assert args.save_period == 10
        assert args.export is True

    def test_defaults(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "d", "--data", "d.yaml", "--model", "m",
        ])
        assert args.epochs == 100
        assert args.batch_size == 16
        assert args.imgsz == 640
        assert args.workers == 4
        assert args.device == "cpu"
        assert args.name is None
        assert args.resume is None
        assert args.seed == 42
        assert args.patience is None
        assert args.optimizer == "Adam"
        assert args.lr == 0.001
        assert args.weight_decay == 0.0001
        assert args.save_period == 5
        assert args.export is False

    def test_requires_dataset(self) -> None:
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["train", "--data", "d.yaml", "--model", "m"])

    def test_requires_data(self) -> None:
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["train", "--dataset", "d", "--model", "m"])

    def test_requires_model(self) -> None:
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["train", "--dataset", "d", "--data", "d.yaml"])

    def test_no_command_shows_help(self) -> None:
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_help_output(self) -> None:
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--help"])


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidateArgs:
    def test_passes_with_valid_yaml_file(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml), "--model", "m",
        ])
        validate_args(args)  # should not raise

    def test_passes_with_data_dir(self, sample_data_dir: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_dir), "--model", "m",
        ])
        validate_args(args)  # should not raise

    def test_fails_missing_data_path(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", "/nonexistent/path", "--model", "m",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_dir_without_data_yaml(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(empty_dir), "--model", "m",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_bad_batch_size(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml),
            "--model", "m", "--batch-size", "0",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_bad_epochs(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml),
            "--model", "m", "--epochs", "0",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_small_imgsz(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml),
            "--model", "m", "--imgsz", "16",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_negative_lr(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml),
            "--model", "m", "--lr", "-0.1",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_fails_bad_device(self, sample_data_yaml: Path) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "t", "--data", str(sample_data_yaml),
            "--model", "m", "--device", "invalid",
        ])
        with pytest.raises(SystemExit):
            validate_args(args)


# ------------------------------------------------------------------
# Configuration builder
# ------------------------------------------------------------------


class TestBuildConfig:
    def test_creates_training_config_data(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "coco2017", "--data", "d.yaml", "--model", "yolo11n.pt",
            "--epochs", "50", "--batch-size", "8", "--lr", "0.01", "--seed", "99",
        ])
        config = _build_config(args)
        assert isinstance(config, TrainingConfigData)
        assert config.model_name == "yolo11n.pt"
        assert config.dataset_version == "coco2017"
        assert config.experiment_name == "coco2017"
        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.learning_rate == 0.01
        assert config.seed == 99
        assert config.image_size == (640, 640)
        assert config.device == "cpu"

    def test_experiment_name_from_name_arg(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "coco2017", "--data", "d.yaml",
            "--model", "m", "--name", "my_exp",
        ])
        config = _build_config(args)
        assert config.experiment_name == "my_exp"
        assert config.dataset_version == "my_exp"

    def test_image_size_tuple(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "d", "--data", "d.yaml", "--model", "m", "--imgsz", "1280",
        ])
        config = _build_config(args)
        assert config.image_size == (1280, 1280)

    def test_optimizer_lowercased(self) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "d", "--data", "d.yaml", "--model", "m", "--optimizer", "SGD",
        ])
        config = _build_config(args)
        assert config.optimizer == "sgd"


# ------------------------------------------------------------------
# Data YAML resolution
# ------------------------------------------------------------------


class TestResolveDataYaml:
    def test_copies_file_to_expected_location(self, tmp_path: Path, sample_data_yaml: Path) -> None:
        original_cwd = Path.cwd()
        try:
            __import__("os").chdir(str(tmp_path))
            result = _resolve_data_yaml("my_exp", sample_data_yaml)
            expected = Path("datasets") / "exports" / "yolo" / "my_exp" / "data.yaml"
            assert result == expected
            assert expected.exists()
            assert expected.read_text() == sample_data_yaml.read_text()
        finally:
            __import__("os").chdir(str(original_cwd))

    def test_handles_directory_with_data_yaml(self, tmp_path: Path, sample_data_dir: Path) -> None:
        original_cwd = Path.cwd()
        try:
            __import__("os").chdir(str(tmp_path))
            result = _resolve_data_yaml("dir_exp", sample_data_dir)
            expected = Path("datasets") / "exports" / "yolo" / "dir_exp" / "data.yaml"
            assert result == expected
            assert expected.exists()
        finally:
            __import__("os").chdir(str(original_cwd))

    def test_raises_on_missing_yaml(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _resolve_data_yaml("bad", empty_dir)


# ------------------------------------------------------------------
# Service factory
# ------------------------------------------------------------------


class TestBuildService:
    def test_returns_training_service(self) -> None:
        service = _build_service("test_exp")
        assert service is not None

    def test_checkpoint_dir_under_artifacts(self) -> None:
        service = _build_service("test_exp")
        cm = service._checkpoint_manager
        expected = str(Path("artifacts") / "training" / "test_exp" / "weights")
        assert expected in str(cm.checkpoints_dir)

    def test_experiment_dir_under_artifacts(self) -> None:
        service = _build_service("test_exp")
        em = service._experiment_manager
        expected = str(Path("artifacts") / "training" / "test_exp")
        assert expected in str(em._experiments_dir)  # type: ignore[attr-defined]

    def test_history_dir_under_artifacts(self) -> None:
        service = _build_service("test_exp")
        hm = service._history_manager
        expected = str(Path("artifacts") / "training" / "test_exp")
        assert expected in str(hm._history_dir)  # type: ignore[attr-defined]

    def test_metrics_dir_under_artifacts(self) -> None:
        service = _build_service("test_exp")
        mm = service._metrics_manager
        expected = str(Path("artifacts") / "training" / "test_exp")
        assert expected in str(mm._metrics_dir)  # type: ignore[attr-defined]

    def test_exports_dir_under_artifacts(self) -> None:
        service = _build_service("test_exp")
        ep = service._export_pipeline
        expected = str(Path("artifacts") / "training" / "test_exp" / "exports")
        assert expected in str(ep._exports_dir)  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------


class TestPrintHeader:
    def test_prints_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        p = build_parser()
        args = p.parse_args([
            "train", "--dataset", "test", "--data", "d.yaml", "--model", "m",
        ])
        _print_header("my_exp", args)
        captured = capsys.readouterr()
        assert "my_exp" in captured.out
        assert "m" in captured.out
        assert "d.yaml" in captured.out


class TestPrintResults:
    def test_prints_training_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = TrainingResult(
            experiment_id="exp_1",
            model_id="mod_1",
            total_epochs_completed=10,
            best_epoch=9,
            best_metric=0.9,
            best_metric_name="mAP50",
            training_duration_seconds=123.0,
            status="completed",
        )
        _print_results(result)
        captured = capsys.readouterr()
        assert "exp_1" in captured.out
        assert "completed" in captured.out
        assert "0.9" in captured.out

    def test_handles_non_training_result(self, caplog: pytest.LogCaptureFixture) -> None:
        _print_results("not a result")
        assert "Unexpected result type" in caplog.text


# ------------------------------------------------------------------
# Module-level entry point via sys.argv
# ------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_help(self) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])

    def test_main_with_valid_args(
        self, tmp_path: Path, sample_data_yaml: Path,
    ) -> None:
        original_cwd = Path.cwd()
        try:
            __import__("os").chdir(str(tmp_path))
            with patch("backend.training.cli.cmd_train") as mock_cmd:
                main([
                    "train", "--dataset", "utest",
                    "--data", str(sample_data_yaml),
                    "--model", "yolo11n.pt",
                ])
                mock_cmd.assert_called_once()
        finally:
            __import__("os").chdir(str(original_cwd))

    def test_main_fails_missing_data(self) -> None:
        with pytest.raises(SystemExit):
            main([
                "train", "--dataset", "utest",
                "--data", "/nonexistent",
                "--model", "yolo11n.pt",
            ])


# ------------------------------------------------------------------
# Integration smoke test — module invocation
# ------------------------------------------------------------------


class TestModuleInvocation:
    _PROJECT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent

    def test_module_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "backend.training", "--help"],
            capture_output=True, text=True, cwd=str(self._PROJECT_DIR),
        )
        assert result.returncode == 0
        assert "DSS Training Platform CLI" in result.stdout

    def test_module_train_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "backend.training", "train", "--help"],
            capture_output=True, text=True, cwd=str(self._PROJECT_DIR),
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout

    def test_module_train_missing_data(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "backend.training", "train",
             "--dataset", "t", "--data", "/nonexistent", "--model", "m"],
            capture_output=True, text=True, cwd=str(self._PROJECT_DIR),
        )
        assert result.returncode == 1
        assert "not found" in result.stderr
