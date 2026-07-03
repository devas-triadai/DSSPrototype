from __future__ import annotations

import pytest

from backend.dataset_conversion.dataset_validator import DatasetValidator
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    GeometryType,
    ImageInfo,
)


class TestDatasetValidator:
    @pytest.fixture
    def validator(self) -> DatasetValidator:
        return DatasetValidator()

    @pytest.mark.asyncio
    async def test_valid_dataset(
        self,
        validator: DatasetValidator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        report = await validator.validate(sample_canonical_dataset)
        assert report.valid

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self,
        validator: DatasetValidator,
    ) -> None:
        ds = CanonicalDataset(name="empty", image_count=0, annotation_count=0, class_count=0)
        report = await validator.validate(ds)
        assert not report.valid
        assert len(report.errors) >= 1

    @pytest.mark.asyncio
    async def test_no_annotations(
        self,
        validator: DatasetValidator,
    ) -> None:
        img = ImageInfo(id="img001", file_path="/path.jpg", width=640, height=480)
        ds = CanonicalDataset(
            name="no_anns",
            images=(img,),
            image_count=1,
            annotation_count=0,
            class_count=0,
        )
        report = await validator.validate(ds)
        assert not report.valid
        assert any("no annotations" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_duplicate_image_ids(
        self,
        validator: DatasetValidator,
    ) -> None:
        img = ImageInfo(id="img001", file_path="/path.jpg", width=640, height=480)
        ds = CanonicalDataset(
            name="dup",
            images=(img, img),
            image_count=2,
            annotation_count=0,
            class_count=0,
        )
        report = await validator.validate(ds)
        assert any("duplicate" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_orphan_annotations(
        self,
        validator: DatasetValidator,
    ) -> None:
        ann = CanonicalAnnotation(
            id="ann001",
            image_id="nonexistent",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=10,
            height=10,
        )
        ds = CanonicalDataset(
            name="orphan",
            annotations=(ann,),
            image_count=0,
            annotation_count=1,
            class_count=1,
        )
        report = await validator.validate(ds)
        assert any("orphan" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_invalid_geometry(
        self,
        validator: DatasetValidator,
    ) -> None:
        ann = CanonicalAnnotation.model_construct(
            id="ann001",
            image_id="img001",
            canonical_label="a.b",
            canonical_name="A",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=0,
            height=-1,
        )
        img = ImageInfo(id="img001", file_path="/path.jpg")
        ds = CanonicalDataset(
            name="bad_geom",
            images=(img,),
            annotations=(ann,),
            image_count=1,
            annotation_count=1,
            class_count=1,
        )
        report = await validator.validate(ds)
        assert any("invalid geometry" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_duplicate_annotation_ids(
        self,
        validator: DatasetValidator,
    ) -> None:
        ann = CanonicalAnnotation(
            id="ann001",
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
            id="ann001",
            image_id="img001",
            canonical_label="c.d",
            canonical_name="C",
            geometry_type=GeometryType.BBOX,
            x=0,
            y=0,
            width=10,
            height=10,
        )
        img = ImageInfo(id="img001", file_path="/path.jpg")
        ds = CanonicalDataset(
            name="dup_anns",
            images=(img,),
            annotations=(ann, ann2),
            image_count=1,
            annotation_count=2,
            class_count=2,
        )
        report = await validator.validate(ds)
        assert any("duplicate" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_images_missing_file_paths(
        self,
        validator: DatasetValidator,
    ) -> None:
        img = ImageInfo.model_construct(id="img001", file_path="")
        ds = CanonicalDataset.model_construct(
            name="no_path",
            images=(img,),
            image_count=1,
            annotation_count=0,
            class_count=0,
        )
        report = await validator.validate(ds)
        assert any("missing" in e.lower() for e in report.errors)

    @pytest.mark.asyncio
    async def test_report_structure(
        self,
        validator: DatasetValidator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        report = await validator.validate(sample_canonical_dataset)
        assert report.total_checks == 8
        assert report.failed_checks == 0
        assert report.passed_checks > 0
