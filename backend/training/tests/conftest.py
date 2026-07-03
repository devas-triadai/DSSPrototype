"""Shared fixtures for training CLI and dataset exporter tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.training.models import TrainingResult

# ------------------------------------------------------------------
# CLI fixtures
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Dataset exporter fixtures
# ------------------------------------------------------------------


def _make_image(img_dir: Path, img_id: str, width: int = 640, height: int = 480) -> Path:
    """Create a minimal valid JPEG image file."""
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{img_id}.jpg"
    # Minimal JPEG file (valid header + minimal content)
    path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
        b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
        b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b"
        b"\x08\x01\xe0\x02\x80\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
        b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04"
        b"\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02"
        b"\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x11"
        b"\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R"
        b"\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
        b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96"
        b"\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
        b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4"
        b"\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
        b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?"
        b"\x00\xf8\xff\xd9"
    )
    return path


@pytest.fixture
def coco_source(tmp_path: Path) -> Path:
    """Create a minimal COCO 2017 dataset structure for testing."""
    source = tmp_path / "coco2017"
    train_img_dir = source / "train2017"
    val_img_dir = source / "val2017"
    ann_dir = source / "annotations"

    train_img_dir.mkdir(parents=True)
    val_img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    # Create sample images
    for i in range(3):
        _make_image(train_img_dir, f"000000{i:06d}")
    for i in range(2):
        _make_image(val_img_dir, f"100000{i:06d}")

    # COCO categories
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "dog", "supercategory": "animal"},
    ]

    # Train annotations
    train_images = [
        {"id": 1, "file_name": "000000000000.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "000000000001.jpg", "width": 640, "height": 480},
        {"id": 3, "file_name": "000000000002.jpg", "width": 640, "height": 480},
    ]
    train_annotations = [
        {"id": 1, "image_id": 1, "category_id": 1,
         "bbox": [100, 150, 200, 300], "area": 60000, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2,
         "bbox": [50, 80, 150, 120], "area": 18000, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 3,
         "bbox": [200, 100, 100, 150], "area": 15000, "iscrowd": 0},
    ]

    train_json = ann_dir / "instances_train2017.json"
    train_json.write_text(
        json.dumps({
            "images": train_images, "annotations": train_annotations,
            "categories": categories,
        }),
        encoding="utf-8",
    )

    # Val annotations
    val_images = [
        {"id": 4, "file_name": "100000000000.jpg", "width": 640, "height": 480},
        {"id": 5, "file_name": "100000000001.jpg", "width": 640, "height": 480},
    ]
    val_annotations = [
        {"id": 4, "image_id": 4, "category_id": 1,
         "bbox": [300, 200, 100, 150], "area": 15000, "iscrowd": 0},
    ]

    val_json = ann_dir / "instances_val2017.json"
    val_json.write_text(
        json.dumps({
            "images": val_images, "annotations": val_annotations,
            "categories": categories,
        }),
        encoding="utf-8",
    )

    return source


@pytest.fixture
def openimages_source(tmp_path: Path) -> Path:
    """Create a minimal Open Images V7 dataset structure for testing."""
    source = tmp_path / "openimages_v7"
    train_dir = source / "train"
    val_dir = source / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for i in range(3):
        _make_image(train_dir, f"train_img_{i:06d}")
    for i in range(2):
        _make_image(val_dir, f"val_img_{i:06d}")

    # Class descriptions
    (source / "class-descriptions-boxable.csv").write_text(
        "/m/0c9q5,Cat\n"
        "/m/0bt9lr,Dog\n"
        "/m/0k4j,Car\n",
        encoding="utf-8",
    )

    # Train annotations CSV
    (source / "train-annotations-bbox.csv").write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
        "train_img_000000,freeform,/m/0c9q5,1,0.1,0.5,0.2,0.6,0,0,0,0,0\n"
        "train_img_000000,freeform,/m/0bt9lr,1,0.3,0.7,0.1,0.8,0,0,0,0,0\n"
        "train_img_000001,freeform,/m/0k4j,1,0.2,0.6,0.3,0.7,0,0,0,0,0\n"
        "train_img_000002,freeform,/m/0c9q5,1,0.0,0.4,0.0,0.5,0,0,0,0,0\n",
        encoding="utf-8",
    )

    # Val annotations CSV
    (source / "val-annotations-bbox.csv").write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
        "val_img_000000,freeform,/m/0bt9lr,1,0.1,0.5,0.2,0.6,0,0,0,0,0\n",
        encoding="utf-8",
    )

    return source


@pytest.fixture
def visdrone_source(tmp_path: Path) -> Path:
    """Create a minimal VisDrone dataset structure for testing."""
    source = tmp_path / "visdrone"

    train_ann = source / "VisDrone2019-DET-train" / "annotations"
    train_img = source / "VisDrone2019-DET-train" / "images"
    val_ann = source / "VisDrone2019-DET-val" / "annotations"
    val_img = source / "VisDrone2019-DET-val" / "images"

    train_img.mkdir(parents=True)
    val_img.mkdir(parents=True)
    train_ann.mkdir(parents=True)
    val_ann.mkdir(parents=True)

    for i in range(3):
        _make_image(train_img, f"{i:06d}")
    for i in range(2):
        _make_image(val_img, f"{i:06d}")

    # VisDrone: bbox_left,bbox_top,bbox_width,bbox_height,score,category,trunc,occ
    (train_ann / "000000.txt").write_text(
        "100,150,200,300,1,2,0,0\n"
        "50,80,150,120,1,4,0,0\n",
        encoding="utf-8",
    )
    (train_ann / "000001.txt").write_text(
        "200,100,100,150,1,3,0,0\n",
        encoding="utf-8",
    )
    (train_ann / "000002.txt").write_text(
        "300,200,50,80,1,1,0,0\n",
        encoding="utf-8",
    )
    (val_ann / "000000.txt").write_text(
        "150,100,300,400,1,2,0,0\n",
        encoding="utf-8",
    )

    return source


@pytest.fixture
def loveda_source(tmp_path: Path) -> Path:
    """Create a minimal LoveDA dataset structure for testing."""
    source = tmp_path / "loveda"

    train_dir = source / "Train"
    val_dir = source / "Val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for i in range(3):
        _make_image(train_dir, f"train_{i:06d}")
    for i in range(2):
        _make_image(val_dir, f"val_{i:06d}")

    # GeoJSON annotations for LoveDA
    geojson_feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"class": "building"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[100, 200], [300, 200], [300, 400], [100, 400], [100, 200]]],
                },
            },
        ],
    }

    (train_dir / "train_000000.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")
    (train_dir / "train_000001.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")
    (val_dir / "val_000000.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")

    return source


@pytest.fixture
def spacenet_source(tmp_path: Path) -> Path:
    """Create a minimal SpaceNet dataset structure for testing."""
    source = tmp_path / "spacenet"

    train_dir = source / "train"
    val_dir = source / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for i in range(3):
        _make_image(train_dir, f"img_{i:06d}")
    for i in range(2):
        _make_image(val_dir, f"img_{i:06d}")

    geojson_feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[50, 100], [250, 100], [250, 300], [50, 300], [50, 100]]],
                },
            },
        ],
    }

    (train_dir / "img_000000.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")
    (train_dir / "img_000001.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")
    (val_dir / "img_000000.geojson").write_text(json.dumps(geojson_feature), encoding="utf-8")

    return source


@pytest.fixture
def seaships_source(tmp_path: Path) -> Path:
    """Create a minimal SeaShips dataset structure for testing."""
    source = tmp_path / "seaships"

    train_dir = source / "train"
    val_dir = source / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    for i in range(3):
        _make_image(train_dir, f"ship_{i:06d}")
    for i in range(2):
        _make_image(val_dir, f"ship_{i:06d}")

    def _make_voc_xml(path: Path, filename: str, bbox: tuple[float, float, float, float]) -> str:
        xmin, ymin, xmax, ymax = bbox
        return (
            f"<annotation>\n"
            f"    <filename>{filename}</filename>\n"
            f"    <size>\n"
            f"        <width>640</width>\n"
            f"        <height>480</height>\n"
            f"        <depth>3</depth>\n"
            f"    </size>\n"
            f"    <object>\n"
            f"        <name>ship</name>\n"
            f"        <bndbox>\n"
            f"            <xmin>{xmin}</xmin>\n"
            f"            <ymin>{ymin}</ymin>\n"
            f"            <xmax>{xmax}</xmax>\n"
            f"            <ymax>{ymax}</ymax>\n"
            f"        </bndbox>\n"
            f"    </object>\n"
            f"</annotation>\n"
        )

    (train_dir / "ship_000000.xml").write_text(
        _make_voc_xml(train_dir, "ship_000000.jpg", (100, 150, 300, 400)),
        encoding="utf-8",
    )
    (train_dir / "ship_000001.xml").write_text(
        _make_voc_xml(train_dir, "ship_000001.jpg", (50, 80, 200, 250)),
        encoding="utf-8",
    )
    (val_dir / "ship_000000.xml").write_text(
        _make_voc_xml(val_dir, "ship_000000.jpg", (150, 200, 400, 450)),
        encoding="utf-8",
    )

    return source


@pytest.fixture(params=[
    "coco", "open_images_v7", "visdrone", "loveda", "spacenet", "seaships",
])
def any_dataset_source(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Parametrized fixture that yields all dataset source types."""
    fixture_map = {
        "coco": coco_source,
        "open_images_v7": openimages_source,
        "visdrone": visdrone_source,
        "loveda": loveda_source,
        "spacenet": spacenet_source,
        "seaships": seaships_source,
    }
    fixture_fn = fixture_map[request.param]
    return fixture_fn(request, tmp_path) if callable(fixture_fn) else fixture_fn
