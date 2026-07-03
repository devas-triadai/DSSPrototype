from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    CoordinateSystem,
    GeometryType,
    ImageInfo,
    MergeConfig,
    SourceAnnotation,
    SourceCategory,
    SplitConfig,
    SplitStrategy,
)


@pytest.fixture
def sample_image() -> ImageInfo:
    return ImageInfo(
        id="img001",
        file_path="/data/images/img001.jpg",
        width=640,
        height=480,
        format="jpg",
        color_space="rgb",
    )


@pytest.fixture
def sample_images() -> list[ImageInfo]:
    return [
        ImageInfo(id=f"img{i:03d}", file_path=f"/data/images/img{i:03d}.jpg", width=640, height=480)
        for i in range(10)
    ]


@pytest.fixture
def sample_source_category_car() -> SourceCategory:
    return SourceCategory(id=1, name="car", supercategory="vehicle")


@pytest.fixture
def sample_source_category_person() -> SourceCategory:
    return SourceCategory(id=2, name="person", supercategory="person")


@pytest.fixture
def sample_source_annotation() -> SourceAnnotation:
    return SourceAnnotation(
        id="ann001",
        image_id="img001",
        category_id=1,
        category_name="car",
        geometry_type=GeometryType.BBOX,
        coordinates=(100.0, 150.0, 200.0, 300.0),
        coordinate_system=CoordinateSystem.PIXEL,
        confidence=0.95,
        image_width=640,
        image_height=480,
    )


@pytest.fixture
def sample_source_annotations() -> list[SourceAnnotation]:
    return [
        SourceAnnotation(
            id=f"ann{i:03d}",
            image_id=f"img{i:03d}",
            category_id=i % 2 + 1,
            category_name="car" if i % 2 == 0 else "person",
            geometry_type=GeometryType.BBOX,
            coordinates=(50.0 + i * 10, 100.0 + i * 5, 100.0, 150.0),
            coordinate_system=CoordinateSystem.PIXEL,
            image_width=640,
            image_height=480,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_canonical_annotation() -> CanonicalAnnotation:
    return CanonicalAnnotation(
        id="canon_ann001",
        image_id="img001",
        canonical_label="ground_vehicle.car",
        canonical_name="Car",
        geometry_type=GeometryType.BBOX,
        x=100.0,
        y=150.0,
        width=200.0,
        height=300.0,
        confidence=0.95,
        source_annotation_id="ann001",
        source_label="car",
    )


@pytest.fixture
def sample_canonical_annotations() -> list[CanonicalAnnotation]:
    return [
        CanonicalAnnotation(
            id=f"canon_ann{i:03d}",
            image_id=f"img{i:03d}",
            canonical_label="ground_vehicle.car" if i % 2 == 0 else "people.person",
            canonical_name="Car" if i % 2 == 0 else "Person",
            geometry_type=GeometryType.BBOX,
            x=50.0 + i * 10,
            y=100.0 + i * 5,
            width=100.0,
            height=150.0,
            confidence=0.95,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_canonical_dataset(
    sample_images: list[ImageInfo],
    sample_canonical_annotations: list[CanonicalAnnotation],
) -> CanonicalDataset:
    return CanonicalDataset(
        id="test-ds-001",
        name="test_dataset",
        images=tuple(sample_images),
        annotations=tuple(sample_canonical_annotations),
        image_count=len(sample_images),
        annotation_count=len(sample_canonical_annotations),
        class_count=2,
        ontology_version="1.0.0",
        pipeline_version="1.0.0",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_datasets=("coco",),
    )


@pytest.fixture
def sample_split_config() -> SplitConfig:
    return SplitConfig(
        strategy=SplitStrategy.RANDOM,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )


@pytest.fixture
def sample_merge_config() -> MergeConfig:
    return MergeConfig(
        strategy="sequential",
        resolve_conflicts=True,
        deduplicate_images=True,
        image_id_prefix="merged",
        annotation_id_prefix="merged_ann",
    )


@pytest.fixture
def coco_json_data() -> str:
    return (
        '{"images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],'
        '"annotations": [{"id": 1, "image_id": 1, "category_id": 1,'
        '"bbox": [10, 20, 100, 200], "area": 20000, "iscrowd": 0}],'
        '"categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}]}'
    )


@pytest.fixture
def yolo_txt_data() -> str:
    return "0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n"


@pytest.fixture
def pascal_voc_xml_data() -> str:
    return """<annotation>
  <filename>img001.jpg</filename>
  <size><width>640</width><height>480</height></size>
  <object><name>car</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>220</ymax></bndbox></object>
  <object><name>person</name><bndbox><xmin>50</xmin><ymin>60</ymin><xmax>150</xmax><ymax>260</ymax></bndbox></object>
</annotation>"""


@pytest.fixture
def open_images_csv_data() -> str:
    h = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax"
    r1 = "img001,freeform,Car,1.0,0.1,0.5,0.2,0.6,0,0,0,0,0"
    r2 = "img002,freeform,Person,0.95,0.2,0.4,0.3,0.7,0,0,0,0,0"
    return f"{h}\n{r1}\n{r2}"
