"""Dataset layout detectors for well-known dataset structures.

Each detector function recognises the official directory layout of a
popular dataset and returns a DatasetLayout on success or None.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from backend.dataset_pipeline.models import DatasetLayout

# ------------------------------------------------------------------
# Detector registry
# ------------------------------------------------------------------

_DETECTORS: list[Callable[[Path], DatasetLayout | None]] = []


def _register(
    fn: Callable[[Path], DatasetLayout | None],
) -> Callable[[Path], DatasetLayout | None]:
    _DETECTORS.append(fn)
    return fn


# ------------------------------------------------------------------
# COCO 2017
# ------------------------------------------------------------------


@_register
def detect_coco2017(root: Path) -> DatasetLayout | None:
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    ann_dir = root / "annotations"
    train_ann = ann_dir / "instances_train2017.json"
    val_ann = ann_dir / "instances_val2017.json"

    if not all(d.is_dir() for d in (train_dir, val_dir, ann_dir)):
        return None
    if not (train_ann.is_file() and val_ann.is_file()):
        return None

    return DatasetLayout(
        dataset_type="coco",
        root_path=root,
        image_directories=[train_dir, val_dir],
        annotation_directory=ann_dir,
        annotation_files=[train_ann, val_ann],
    )


# ------------------------------------------------------------------
# Open Images V7
# ------------------------------------------------------------------


@_register
def detect_open_images_v7(root: Path) -> DatasetLayout | None:
    subdirs = ["train", "validation", "test"]
    expected = [root / d for d in subdirs]
    if not all(d.is_dir() for d in expected):
        return None

    return DatasetLayout(
        dataset_type="open_images_v7",
        root_path=root,
        image_directories=expected,
        annotation_directory=root,
        annotation_files=[],
    )


# ------------------------------------------------------------------
# VisDrone
# ------------------------------------------------------------------


@_register
def detect_visdrone(root: Path) -> DatasetLayout | None:
    subdirs = ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]
    expected = [root / d for d in subdirs]
    if not all(d.is_dir() for d in expected):
        return None

    return DatasetLayout(
        dataset_type="visdrone",
        root_path=root,
        image_directories=[d / "images" if (d / "images").is_dir() else d for d in expected],
        annotation_directory=root,
        annotation_files=[],
    )


# ------------------------------------------------------------------
# LoveDA
# ------------------------------------------------------------------


@_register
def detect_loveda(root: Path) -> DatasetLayout | None:
    subdirs = ["Train", "Val", "Test"]
    expected = [root / d for d in subdirs]
    if not all(d.is_dir() for d in expected):
        return None

    return DatasetLayout(
        dataset_type="loveda",
        root_path=root,
        image_directories=expected,
        annotation_directory=root,
        annotation_files=[],
    )


# ------------------------------------------------------------------
# SpaceNet
# ------------------------------------------------------------------


@_register
def detect_spacenet(root: Path) -> DatasetLayout | None:
    aoi_dirs = sorted(root.glob("AOI_*"))
    if not aoi_dirs:
        return None

    return DatasetLayout(
        dataset_type="spacenet",
        root_path=root,
        image_directories=aoi_dirs,
        annotation_directory=root,
        annotation_files=[],
    )


# ------------------------------------------------------------------
# SeaShips
# ------------------------------------------------------------------


@_register
def detect_seaships(root: Path) -> DatasetLayout | None:
    subdirs = ["train", "val"]
    expected = [root / d for d in subdirs]
    if not all(d.is_dir() for d in expected):
        return None

    return DatasetLayout(
        dataset_type="seaships",
        root_path=root,
        image_directories=expected,
        annotation_directory=root,
        annotation_files=[],
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def detect_layout(source_path: Path) -> DatasetLayout | None:
    """Try each registered detector and return the first match."""
    resolved = source_path.resolve() if source_path.is_absolute() else source_path
    if not resolved.is_dir():
        return None
    for detector in _DETECTORS:
        layout = detector(resolved)
        if layout is not None:
            return layout
    return None
