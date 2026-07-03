"""Shared fixtures for training CLI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.training.models import TrainingResult


@pytest.fixture
def sample_data_yaml(tmp_path: Path) -> Path:
    """Create a minimal YOLO data.yaml."""
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: /data/coco\n"
        "train: train2017\n"
        "val: val2017\n"
        "nc: 1\n"
        "names: ['person']\n",
    )
    return yaml_path


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a directory containing data.yaml."""
    data_dir = tmp_path / "coco_dataset"
    data_dir.mkdir(exist_ok=True)
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: train\n"
        "val: val\n"
        "nc: 1\n"
        "names: ['person']\n",
    )
    return data_dir


@pytest.fixture
def mock_training_result() -> MagicMock:
    return MagicMock(
        spec=TrainingResult,
        experiment_id="exp_001",
        model_id="model_001",
        total_epochs_completed=50,
        best_epoch=48,
        best_metric=0.85,
        best_metric_name="mAP50",
        training_duration_seconds=3600.0,
        status="completed",
        final_metrics=MagicMock(
            mAP50=0.85,
            mAP50_95=0.62,
        ),
    )
