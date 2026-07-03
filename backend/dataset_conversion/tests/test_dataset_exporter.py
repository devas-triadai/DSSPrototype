from __future__ import annotations

import json
import os
import tempfile

import pytest

from backend.dataset_conversion.dataset_exporter import DatasetExporter
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    GeometryType,
    ImageInfo,
)


class TestDatasetExporter:
    @pytest.fixture
    def exporter(self) -> DatasetExporter:
        return DatasetExporter()

    @pytest.fixture
    def yolo_ds(self) -> CanonicalDataset:
        img = ImageInfo(id="img001", file_path="/path.jpg", width=640, height=480)
        ann = CanonicalAnnotation(
            id="ann001",
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="Car",
            geometry_type=GeometryType.BBOX,
            x=100,
            y=100,
            width=200,
            height=300,
            confidence=1.0,
        )
        return CanonicalDataset(
            name="yolo_test",
            images=(img,),
            annotations=(ann,),
            image_count=1,
            annotation_count=1,
            class_count=1,
        )

    @pytest.mark.asyncio
    async def test_export_canonical(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporter.export(sample_canonical_dataset, "canonical", tmpdir)
            assert result.images_exported == 10
            assert result.annotations_exported == 10
            assert result.file_count >= 1
            json_path = os.path.join(tmpdir, "dataset.json")
            assert os.path.isfile(json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert data["info"]["dataset_name"] == "test_dataset"
            assert len(data["images"]) == 10
            assert len(data["annotations"]) == 10
            assert len(data["categories"]) >= 2

    @pytest.mark.asyncio
    async def test_export_coco(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporter.export(sample_canonical_dataset, "coco_json", tmpdir)
            assert result.export_format == "coco_json"
            json_path = os.path.join(tmpdir, "coco_annotations.json")
            assert os.path.isfile(json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert "info" in data
            assert "images" in data
            assert "annotations" in data
            assert "categories" in data

    @pytest.mark.asyncio
    async def test_export_yolo(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporter.export(sample_canonical_dataset, "yolo_txt", tmpdir)
            assert result.export_format == "yolo_txt"
            classes_path = os.path.join(tmpdir, "classes.txt")
            assert os.path.isfile(classes_path)
            labels_dir = os.path.join(tmpdir, "labels")
            assert os.path.isdir(labels_dir)
            label_files = os.listdir(labels_dir)
            assert len(label_files) > 0

    @pytest.mark.asyncio
    async def test_export_yolo_content(
        self,
        exporter: DatasetExporter,
        yolo_ds: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await exporter.export(yolo_ds, "yolo_txt", tmpdir)
            label_file = os.path.join(tmpdir, "labels", "img001.txt")
            with open(label_file) as f:
                content = f.read().strip()
            assert len(content) > 0
            parts = content.split()
            assert len(parts) == 5

    @pytest.mark.asyncio
    async def test_export_unsupported_format(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported export format"):
            await exporter.export(sample_canonical_dataset, "unknown", "/tmp")

    @pytest.mark.asyncio
    async def test_supported_export_formats(
        self,
        exporter: DatasetExporter,
    ) -> None:
        formats = await exporter.supported_export_formats()
        assert "canonical" in formats
        assert "coco_json" in formats
        assert "yolo_txt" in formats

    @pytest.mark.asyncio
    async def test_export_canonical_creates_directory(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "output")
            result = await exporter.export(sample_canonical_dataset, "canonical", nested)
            assert os.path.isdir(nested)
            assert result.file_count == 1

    @pytest.mark.asyncio
    async def test_export_yolo_empty_dataset(
        self,
        exporter: DatasetExporter,
    ) -> None:
        ds = CanonicalDataset(name="empty", image_count=0, annotation_count=0, class_count=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await exporter.export(ds, "yolo_txt", tmpdir)
            assert result.images_exported == 0

    @pytest.mark.asyncio
    async def test_export_canonical_contains_correct_keys(
        self,
        exporter: DatasetExporter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            await exporter.export(sample_canonical_dataset, "canonical", tmpdir)
            with open(os.path.join(tmpdir, "dataset.json")) as f:
                data = json.load(f)
            assert "info" in data
            assert "images" in data
            assert "annotations" in data
            assert "categories" in data
            assert data["info"]["ontology_version"] == "1.0.0"
            assert data["info"]["pipeline_version"] == "1.0.0"
