from __future__ import annotations

import pytest

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    GeometryType,
    ImageInfo,
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
        file_size_bytes=102400,
    )


@pytest.fixture
def sample_images() -> list[ImageInfo]:
    return [
        ImageInfo(
            id="img001",
            file_path="/data/images/img001.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
        ),
        ImageInfo(
            id="img002",
            file_path="/data/images/img002.jpg",
            width=1280,
            height=720,
            format="png",
            color_space="rgb",
        ),
        ImageInfo(
            id="img003",
            file_path="/data/images/img003.jpg",
            width=320,
            height=240,
            format="jpg",
            color_space="rgb",
        ),
        ImageInfo(
            id="img004",
            file_path="/data/images/img004.jpg",
            width=1920,
            height=1080,
            format="webp",
            color_space="rgb",
        ),
        ImageInfo(
            id="img005",
            file_path="/data/images/img005.jpg",
            width=100,
            height=50,
            format="jpg",
            color_space="rgb",
        ),
    ]


@pytest.fixture
def sample_tiny_image() -> ImageInfo:
    return ImageInfo(
        id="img_tiny",
        file_path="/data/images/tiny.jpg",
        width=16,
        height=16,
        format="jpg",
        color_space="rgb",
    )


@pytest.fixture
def sample_large_image() -> ImageInfo:
    return ImageInfo(
        id="img_huge",
        file_path="/data/images/huge.jpg",
        width=20000,
        height=20000,
        format="jpg",
        color_space="rgb",
    )


@pytest.fixture
def sample_wrong_format_image() -> ImageInfo:
    return ImageInfo(
        id="img_bmp",
        file_path="/data/images/img.bmp",
        width=640,
        height=480,
        format="bmp",
        color_space="rgb",
    )


@pytest.fixture
def sample_wrong_color_image() -> ImageInfo:
    return ImageInfo(
        id="img_gray",
        file_path="/data/images/gray.png",
        width=640,
        height=480,
        format="png",
        color_space="grayscale",
    )


@pytest.fixture
def sample_annotations() -> list[CanonicalAnnotation]:
    return [
        CanonicalAnnotation(
            id="ann001",
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=100.0,
            y=150.0,
            width=200.0,
            height=300.0,
            confidence=0.95,
        ),
        CanonicalAnnotation(
            id="ann002",
            image_id="img001",
            canonical_label="people.person",
            canonical_name="person",
            geometry_type=GeometryType.BBOX,
            x=50.0,
            y=50.0,
            width=80.0,
            height=120.0,
            confidence=0.90,
        ),
        CanonicalAnnotation(
            id="ann003",
            image_id="img002",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=200.0,
            y=300.0,
            width=150.0,
            height=100.0,
            confidence=0.85,
        ),
        CanonicalAnnotation(
            id="ann004",
            image_id="img002",
            canonical_label="people.person",
            canonical_name="person",
            geometry_type=GeometryType.BBOX,
            x=400.0,
            y=100.0,
            width=60.0,
            height=90.0,
            confidence=0.92,
        ),
        CanonicalAnnotation(
            id="ann005",
            image_id="img003",
            canonical_label="ground_vehicle.truck",
            canonical_name="truck",
            geometry_type=GeometryType.BBOX,
            x=300.0,
            y=200.0,
            width=250.0,
            height=350.0,
            confidence=0.88,
        ),
        CanonicalAnnotation(
            id="ann006",
            image_id="img003",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=10.0,
            y=20.0,
            width=100.0,
            height=80.0,
            confidence=0.95,
        ),
        CanonicalAnnotation(
            id="ann007",
            image_id="img004",
            canonical_label="people.person",
            canonical_name="person",
            geometry_type=GeometryType.POLYGON,
            x=0.0,
            y=0.0,
            width=50.0,
            height=100.0,
            confidence=0.70,
        ),
        CanonicalAnnotation(
            id="ann008",
            image_id="img005",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=15.0,
            y=10.0,
            width=30.0,
            height=20.0,
            confidence=0.99,
        ),
        CanonicalAnnotation(
            id="ann009",
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=500.0,
            y=400.0,
            width=100.0,
            height=80.0,
            confidence=0.80,
        ),
        CanonicalAnnotation(
            id="ann010",
            image_id="img005",
            canonical_label="people.person",
            canonical_name="person",
            geometry_type=GeometryType.BBOX,
            x=5.0,
            y=5.0,
            width=40.0,
            height=30.0,
            confidence=0.75,
        ),
    ]


@pytest.fixture
def sample_invalid_annotations() -> list[CanonicalAnnotation]:
    return [
        CanonicalAnnotation(
            id="neg_coord",
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=-10.0,
            y=-20.0,
            width=100.0,
            height=80.0,
        ),
        CanonicalAnnotation(
            id="zero_area",
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=10.0,
            y=10.0,
            width=0.0,
            height=0.0,
        ),
        CanonicalAnnotation(
            id="out_of_bounds",
            image_id="img001",
            canonical_label="people.person",
            canonical_name="person",
            geometry_type=GeometryType.BBOX,
            x=600.0,
            y=450.0,
            width=100.0,
            height=100.0,
        ),
        CanonicalAnnotation(
            id="neg_area",
            image_id="img001",
            canonical_label="ground_vehicle.truck",
            canonical_name="truck",
            geometry_type=GeometryType.BBOX,
            x=100.0,
            y=100.0,
            width=-50.0,
            height=-30.0,
        ),
    ]


@pytest.fixture
def sample_dataset(
    sample_images: list[ImageInfo],
    sample_annotations: list[CanonicalAnnotation],
) -> CanonicalDataset:
    return CanonicalDataset(
        name="test_dataset",
        images=tuple(sample_images),
        annotations=tuple(sample_annotations),
        image_count=len(sample_images),
        annotation_count=len(sample_annotations),
        class_count=3,
        ontology_version="1.0.0",
        pipeline_version="1.0.0",
    )


@pytest.fixture
def sample_empty_dataset() -> CanonicalDataset:
    return CanonicalDataset(
        name="empty_dataset",
        images=(),
        annotations=(),
        image_count=0,
        annotation_count=0,
        class_count=0,
        ontology_version="1.0.0",
        pipeline_version="1.0.0",
    )


@pytest.fixture
def ontology_classes() -> list[str]:
    return [
        "ground_vehicle.car",
        "ground_vehicle.truck",
        "ground_vehicle.bus",
        "people.person",
        "people.cyclist",
        "aerial.drone",
    ]


@pytest.fixture
def sample_duplicate_dataset() -> CanonicalDataset:
    images = (
        ImageInfo(
            id="img001",
            file_path="/data/a.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
        ),
        ImageInfo(
            id="img002",
            file_path="/data/b.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
        ),
    )
    annotations = (
        CanonicalAnnotation(
            id="ann001",
            image_id="img001",
            canonical_label="car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=0.0,
            y=0.0,
            width=10.0,
            height=10.0,
        ),
        CanonicalAnnotation(
            id="ann002",
            image_id="img001",
            canonical_label="car",
            canonical_name="car",
            geometry_type=GeometryType.BBOX,
            x=0.0,
            y=0.0,
            width=10.0,
            height=10.0,
        ),
    )
    return CanonicalDataset(
        name="duplicate_dataset",
        images=images,
        annotations=annotations,
        image_count=2,
        annotation_count=2,
        class_count=1,
        ontology_version="1.0.0",
        pipeline_version="1.0.0",
    )
