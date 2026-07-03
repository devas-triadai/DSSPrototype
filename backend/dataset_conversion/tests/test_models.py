from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    CoordinateSystem,
    DatasetFormat,
    DatasetManifest,
    DatasetStatistics,
    ExportResult,
    GeometryType,
    ImageInfo,
    LoadResult,
    MergeConfig,
    MergeResult,
    SourceAnnotation,
    SourceCategory,
    SplitConfig,
    SplitResult,
    SplitStrategy,
    SplitType,
    ValidationReport,
)


class TestImageInfo:
    def test_minimal(self) -> None:
        img = ImageInfo(id="test", file_path="/path/to/img.jpg")
        assert img.id == "test"
        assert img.width is None
        assert img.height is None
        assert img.format is None
        assert img.color_space is None
        assert img.file_size_bytes is None
        assert img.metadata == {}

    def test_full(self) -> None:
        img = ImageInfo(
            id="test",
            file_path="/path/to/img.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
            file_size_bytes=1024,
            metadata={"source": "coco"},
        )
        assert img.width == 640
        assert img.height == 480
        assert img.file_size_bytes == 1024

    def test_frozen(self) -> None:
        img = ImageInfo(id="test", file_path="/path.jpg")
        with pytest.raises(Exception):
            img.id = "changed"


class TestSourceAnnotation:
    def test_minimal(self) -> None:
        ann = SourceAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.BBOX,
            coordinates=(1, 2, 3, 4),
        )
        assert ann.coordinate_system == CoordinateSystem.PIXEL
        assert ann.confidence is None

    def test_with_confidence(self) -> None:
        ann = SourceAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.BBOX,
            coordinates=(1, 2, 3, 4),
            confidence=0.95,
        )
        assert ann.confidence == 0.95

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            SourceAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                category_name="car",
                geometry_type=GeometryType.BBOX,
                coordinates=(1, 2, 3, 4),
                confidence=1.5,
            )

    def test_empty_category_name(self) -> None:
        with pytest.raises(ValidationError):
            SourceAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                category_name="",
                geometry_type=GeometryType.BBOX,
                coordinates=(1, 2, 3, 4),
            )

    def test_frozen(self) -> None:
        ann = SourceAnnotation(
            id=1,
            image_id=1,
            category_id=1,
            category_name="car",
            geometry_type=GeometryType.BBOX,
            coordinates=(1, 2, 3, 4),
        )
        with pytest.raises(Exception):
            ann.id = 999


class TestCanonicalAnnotation:
    def test_minimal(self) -> None:
        ann = CanonicalAnnotation(
            image_id="img001",
            canonical_label="ground_vehicle.car",
            canonical_name="Car",
            geometry_type=GeometryType.BBOX,
            x=10,
            y=20,
            width=100,
            height=200,
        )
        assert ann.confidence == 1.0
        assert ann.id is not None
        assert ann.source_annotation_id is None

    def test_with_all_fields(self) -> None:
        ann = CanonicalAnnotation(
            id="custom_id",
            image_id="img001",
            canonical_label="people.person",
            canonical_name="Person",
            geometry_type=GeometryType.BBOX,
            x=10,
            y=20,
            width=100,
            height=200,
            confidence=0.8,
            source_annotation_id="ann001",
            source_label="person",
            metadata={"pose": "standing"},
        )
        assert ann.id == "custom_id"
        assert ann.metadata == {"pose": "standing"}

    def test_auto_generates_id(self) -> None:
        ann1 = CanonicalAnnotation(
            image_id="img001",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=10,
            height=10,
        )
        ann2 = CanonicalAnnotation(
            image_id="img001",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=10,
            height=10,
        )
        assert ann1.id != ann2.id


class TestSourceCategory:
    def test_minimal(self) -> None:
        cat = SourceCategory(id=1, name="car")
        assert cat.supercategory is None

    def test_with_supercategory(self) -> None:
        cat = SourceCategory(id=1, name="car", supercategory="vehicle")
        assert cat.supercategory == "vehicle"

    def test_string_id(self) -> None:
        cat = SourceCategory(id="/m/012345", name="car")
        assert cat.id == "/m/012345"


class TestCanonicalDataset:
    def test_minimal(self, sample_canonical_dataset: CanonicalDataset) -> None:
        ds = sample_canonical_dataset
        assert ds.name == "test_dataset"
        assert ds.image_count == 10
        assert ds.annotation_count == 10
        assert ds.class_count == 2

    def test_auto_generates_id(self) -> None:
        ds1 = CanonicalDataset(name="a", image_count=0, annotation_count=0, class_count=0)
        ds2 = CanonicalDataset(name="a", image_count=0, annotation_count=0, class_count=0)
        assert ds1.id != ds2.id

    def test_frozen(self, sample_canonical_dataset: CanonicalDataset) -> None:
        with pytest.raises(Exception):
            sample_canonical_dataset.name = "changed"


class TestEnums:
    def test_geometry_type_values(self) -> None:
        assert GeometryType.BBOX.value == "bbox"
        assert GeometryType.POLYGON.value == "polygon"
        assert GeometryType.OBB.value == "obb"

    def test_coordinate_system_values(self) -> None:
        assert CoordinateSystem.PIXEL.value == "pixel"
        assert CoordinateSystem.NORMALIZED.value == "normalized"

    def test_dataset_format_values(self) -> None:
        assert DatasetFormat.COCO_JSON.value == "coco_json"
        assert DatasetFormat.YOLO_TXT.value == "yolo_txt"

    def test_split_type_values(self) -> None:
        assert SplitType.TRAIN.value == "train"
        assert SplitType.VAL.value == "val"
        assert SplitType.TEST.value == "test"

    def test_split_strategy_values(self) -> None:
        assert SplitStrategy.RANDOM.value == "random"
        assert SplitStrategy.STRATIFIED.value == "stratified"


class TestSplitConfig:
    def test_defaults(self) -> None:
        cfg = SplitConfig()
        assert cfg.strategy == SplitStrategy.RANDOM
        assert cfg.train_ratio == 0.7
        assert cfg.seed == 42
        assert cfg.shuffle is True

    def test_custom(self) -> None:
        cfg = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=123,
            shuffle=False,
        )
        assert cfg.strategy == SplitStrategy.STRATIFIED


class TestMergeConfig:
    def test_defaults(self) -> None:
        cfg = MergeConfig()
        assert cfg.strategy == "sequential"
        assert cfg.deduplicate_images is True


class TestValidationReport:
    def test_valid(self) -> None:
        report = ValidationReport(
            valid=True,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
        )
        assert report.valid

    def test_invalid(self) -> None:
        report = ValidationReport(
            valid=False,
            total_checks=5,
            passed_checks=3,
            failed_checks=2,
            errors=("error1", "error2"),
        )
        assert not report.valid
        assert len(report.errors) == 2


class TestDatasetStatistics:
    def test_minimal(self) -> None:
        stats = DatasetStatistics(
            total_images=10,
            total_annotations=50,
            total_classes=3,
            images_with_annotations=8,
            images_without_annotations=2,
            avg_annotations_per_image=5.0,
            min_annotations_per_image=0,
            max_annotations_per_image=15,
            avg_image_width=640.0,
            avg_image_height=480.0,
            avg_bbox_width=100.0,
            avg_bbox_height=150.0,
        )
        assert stats.total_images == 10
        assert stats.class_balance == {}


class TestDatasetManifest:
    def test_minimal(self) -> None:
        manifest = DatasetManifest(
            dataset_version="1.0.0",
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        assert manifest.manifest_id is not None
        assert manifest.source_datasets == ()

    def test_auto_timestamp(self) -> None:
        manifest = DatasetManifest(
            dataset_version="1.0.0",
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        assert manifest.conversion_timestamp is not None


class TestExportResult:
    def test_minimal(self) -> None:
        result = ExportResult(
            export_format="coco_json",
            output_path="/output",
            images_exported=10,
            annotations_exported=50,
            file_count=1,
            file_size_bytes=1024,
        )
        assert result.annotations_exported == 50


class TestLoadResult:
    def test_empty(self) -> None:
        result = LoadResult(
            dataset_name="test",
            source_path="/path",
            dataset_format="coco_json",
            image_count=0,
            annotation_count=0,
            category_count=0,
        )
        assert result.images == ()
        assert result.annotations == ()


class TestSplitResult:
    def test_minimal(
        self,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = SplitResult(
            train=sample_canonical_dataset,
            val=sample_canonical_dataset,
            test=sample_canonical_dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        assert result.train.name == "test_dataset"


class TestMergeResult:
    def test_minimal(
        self,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = MergeResult(
            dataset=sample_canonical_dataset,
            total_images=10,
            total_annotations=10,
        )
        assert result.deduplicated_count == 0


class TestCanonicalAnnotationGeomValidation:
    def test_negative_width(self) -> None:
        with pytest.raises(ValidationError):
            CanonicalAnnotation(
                image_id="img001",
                canonical_label="a.b",
                canonical_name="A",
                geometry_type=GeometryType.BBOX,
                x=0,
                y=0,
                width=-1,
                height=10,
            )

    def test_zero_width_ok(self) -> None:
        ann = CanonicalAnnotation(
            image_id="img001",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=0,
            height=10,
        )
        assert ann.width == 0

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CanonicalAnnotation(
                image_id="img001",
                canonical_label="a.b",
                canonical_name="A",
                geometry_type=GeometryType.BBOX,
                x=0,
                y=0,
                width=10,
                height=10,
                confidence=1.5,
            )

    def test_annotation_metadata_default(self) -> None:
        ann = CanonicalAnnotation(
            image_id="img001",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=10,
            height=10,
        )
        assert ann.metadata == {}


class TestCanonicalDatasetEdgeCases:
    def test_empty_dataset(self) -> None:
        ds = CanonicalDataset(name="empty", image_count=0, annotation_count=0, class_count=0)
        assert ds.images == ()
        assert ds.annotations == ()
        assert ds.source_datasets == ()

    def test_dataset_with_string_id(self) -> None:
        ds = CanonicalDataset(
            id="custom-id",
            name="custom",
            image_count=1,
            annotation_count=1,
            class_count=1,
        )
        assert ds.id == "custom-id"

    def test_dataset_timestamps(self) -> None:
        ds = CanonicalDataset(name="ts", image_count=0, annotation_count=0, class_count=0)
        assert ds.created_at is not None


class TestSplitConfigEdgeCases:
    def test_zero_val_ratio(self) -> None:
        cfg = SplitConfig(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=42)
        assert cfg.val_ratio == 0.0

    def test_stratify_by(self) -> None:
        cfg = SplitConfig(strategy=SplitStrategy.STRATIFIED, stratify_by="class")
        assert cfg.stratify_by == "class"

    def test_no_shuffle(self) -> None:
        cfg = SplitConfig(shuffle=False)
        assert cfg.shuffle is False


class TestMergeConfigEdgeCases:
    def test_deduplicate_off(self) -> None:
        cfg = MergeConfig(deduplicate_images=False)
        assert cfg.deduplicate_images is False


class TestValidationReportEdgeCases:
    def test_no_warnings(self) -> None:
        r = ValidationReport(valid=True, total_checks=1, passed_checks=1, failed_checks=0)
        assert r.warnings == ()

    def test_with_warnings(self) -> None:
        r = ValidationReport(
            valid=True,
            total_checks=1,
            passed_checks=1,
            failed_checks=0,
            warnings=("low coverage",),
        )
        assert "low coverage" in r.warnings


class TestDatasetStatisticsEdgeCases:
    def test_zero_totals(self) -> None:
        stats = DatasetStatistics(
            total_images=0,
            total_annotations=0,
            total_classes=0,
            images_with_annotations=0,
            images_without_annotations=0,
            avg_annotations_per_image=0.0,
            min_annotations_per_image=0,
            max_annotations_per_image=0,
            avg_image_width=0.0,
            avg_image_height=0.0,
            avg_bbox_width=0.0,
            avg_bbox_height=0.0,
        )
        assert stats.class_balance == {}

    def test_with_classes(self) -> None:
        stats = DatasetStatistics(
            total_images=10,
            total_annotations=30,
            total_classes=2,
            images_with_annotations=8,
            images_without_annotations=2,
            avg_annotations_per_image=3.0,
            min_annotations_per_image=0,
            max_annotations_per_image=10,
            avg_image_width=640.0,
            avg_image_height=480.0,
            avg_bbox_width=100.0,
            avg_bbox_height=100.0,
            classes=(("class_a", 20), ("class_b", 10)),
            class_balance={"class_a": 0.666, "class_b": 0.333},
        )
        assert stats.classes[0][1] == 20


class TestSourceCategoryEdgeCases:
    def test_integer_name(self) -> None:
        cat = SourceCategory(id=0, name="0")
        assert cat.name == "0"

    def test_string_name_with_spaces(self) -> None:
        cat = SourceCategory(id=1, name="passenger car")
        assert cat.name == "passenger car"


class TestImageInfoEdgeCases:
    def test_zero_dimensions(self) -> None:
        img = ImageInfo.model_construct(id="test", file_path="/path.jpg", width=0, height=0)
        assert img.width == 0
        assert img.height == 0

    def test_large_file_size(self) -> None:
        img = ImageInfo(id="test", file_path="/path.jpg", file_size_bytes=2**31)
        assert img.file_size_bytes == 2**31
