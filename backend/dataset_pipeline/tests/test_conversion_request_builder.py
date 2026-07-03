"""Tests for ConversionRequestBuilder.

Covers all supported layout types, fallback paths, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.dataset_pipeline.conversion_request_builder import (
    ConversionRequest,
    ConversionRequestBuilder,
)
from backend.dataset_pipeline.models import DatasetLayout


class TestBuildCoco:
    def test_with_full_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="coco",
            root_path=Path("/data/coco"),
            image_directories=[Path("/data/coco/train2017"), Path("/data/coco/val2017")],
            annotation_directory=Path("/data/coco/annotations"),
            annotation_files=[Path("/data/coco/annotations/instances_train2017.json")],
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/coco/annotations/instances_train2017.json")
        assert request.dataset_format == "coco_json"
        assert Path(request.kwargs["data_dir"]) == Path("/data/coco/train2017")

    def test_with_minimal_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="coco",
            root_path=Path("/data/coco"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert request.source_path.endswith("instances_train2017.json")
        assert request.dataset_format == "coco_json"
        assert request.kwargs["data_dir"].endswith("train2017")


class TestBuildOpenImagesV7:
    def test_with_image_directories(self) -> None:
        layout = DatasetLayout(
            dataset_type="open_images_v7",
            root_path=Path("/data/oi"),
            image_directories=[Path("/data/oi/train")],
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/oi/train-annotations-bbox.csv")
        assert request.dataset_format == "open_images_csv"
        assert Path(request.kwargs["data_dir"]) == Path("/data/oi/train")

    def test_fallback_data_dir(self) -> None:
        layout = DatasetLayout(
            dataset_type="open_images_v7",
            root_path=Path("/data/oi"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert request.kwargs["data_dir"].endswith("train")


class TestBuildVisdrone:
    def test_with_annotation_directory(self) -> None:
        layout = DatasetLayout(
            dataset_type="visdrone",
            root_path=Path("/data/visdrone"),
            image_directories=[Path("/data/visdrone/VisDrone2019-DET-train/images")],
            annotation_directory=Path("/data/visdrone/VisDrone2019-DET-train/annotations"),
        )
        request = ConversionRequestBuilder().build(layout)
        expected_ann = Path("/data/visdrone/VisDrone2019-DET-train/annotations")
        assert Path(request.source_path) == expected_ann
        assert request.dataset_format == "coco_json"
        expected_img = Path("/data/visdrone/VisDrone2019-DET-train/images")
        assert Path(request.kwargs["data_dir"]) == expected_img

    def test_fallback_paths(self) -> None:
        layout = DatasetLayout(
            dataset_type="visdrone",
            root_path=Path("/data/visdrone"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert "annotations" in request.source_path
        assert "images" in request.kwargs["data_dir"]


class TestBuildLoveda:
    def test_with_annotation_file(self) -> None:
        layout = DatasetLayout(
            dataset_type="loveda",
            root_path=Path("/data/loveda"),
            image_directories=[Path("/data/loveda/Train")],
            annotation_files=[Path("/data/loveda/Train/annotations.geojson")],
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/loveda/Train/annotations.geojson")
        assert request.dataset_format == "coco_json"
        assert Path(request.kwargs["data_dir"]) == Path("/data/loveda/Train")

    def test_fallback_root(self) -> None:
        layout = DatasetLayout(
            dataset_type="loveda",
            root_path=Path("/data/loveda"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/loveda")
        assert request.kwargs["data_dir"].endswith("Train")


class TestBuildSpacenet:
    def test_with_annotation_file(self) -> None:
        layout = DatasetLayout(
            dataset_type="spacenet",
            root_path=Path("/data/spacenet"),
            image_directories=[Path("/data/spacenet/AOI_2_Vegas")],
            annotation_files=[Path("/data/spacenet/AOI_2_Vegas/geometries.geojson")],
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/spacenet/AOI_2_Vegas/geometries.geojson")
        assert request.dataset_format == "coco_json"
        assert Path(request.kwargs["data_dir"]) == Path("/data/spacenet/AOI_2_Vegas")

    def test_fallback_root(self) -> None:
        layout = DatasetLayout(
            dataset_type="spacenet",
            root_path=Path("/data/spacenet"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/spacenet")
        assert request.kwargs["data_dir"].endswith("AOI_2_Vegas")


class TestBuildSeaships:
    def test_with_annotation_files(self) -> None:
        layout = DatasetLayout(
            dataset_type="seaships",
            root_path=Path("/data/seaships"),
            image_directories=[Path("/data/seaships/train")],
            annotation_files=[Path("/data/seaships/train/000001.xml")],
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/seaships/train/000001.xml")
        assert request.dataset_format == "coco_json"
        assert Path(request.kwargs["data_dir"]) == Path("/data/seaships/train")

    def test_fallback_root(self) -> None:
        layout = DatasetLayout(
            dataset_type="seaships",
            root_path=Path("/data/seaships"),
        )
        request = ConversionRequestBuilder().build(layout)
        assert Path(request.source_path) == Path("/data/seaships")
        assert request.kwargs["data_dir"].endswith("train")


class TestConversionRequestModel:
    def test_default_kwargs_is_empty(self) -> None:
        req = ConversionRequest(source_path="/data/test", dataset_format="coco_json")
        assert req.kwargs == {}

    def test_with_kwargs(self) -> None:
        req = ConversionRequest(
            source_path="/data/test/ann.json",
            dataset_format="coco_json",
            kwargs={"data_dir": "/data/test/images"},
        )
        assert req.source_path == "/data/test/ann.json"
        assert req.dataset_format == "coco_json"
        assert req.kwargs == {"data_dir": "/data/test/images"}


class TestBuildErrors:
    def test_unknown_dataset_type(self) -> None:
        layout = DatasetLayout(
            dataset_type="unknown_format_123",
            root_path=Path("/data/unknown"),
        )
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            ConversionRequestBuilder().build(layout)

    def test_empty_dataset_type(self) -> None:
        layout = DatasetLayout(
            dataset_type="",
            root_path=Path("/data/empty"),
        )
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            ConversionRequestBuilder().build(layout)
