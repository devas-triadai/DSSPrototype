from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.dataset_validator import DatasetValidator


class TestDatasetValidator:
    @pytest.fixture
    def validator(self) -> DatasetValidator:
        return DatasetValidator()

    @pytest.mark.asyncio
    async def test_valid_dataset(
        self, validator: DatasetValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset)
        assert result["has_images"] is True
        assert result["has_annotations"] is True
        assert result["has_classes"] is True
        assert result["all_references_valid"] is True

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, validator: DatasetValidator, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_empty_dataset)
        assert result["has_images"] is False
        assert result["has_annotations"] is False
        assert result["has_classes"] is False

    @pytest.mark.asyncio
    async def test_broken_references(self, validator: DatasetValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        ds = CanonicalDataset(
            name="broken",
            images=(ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),),
            annotations=(
                CanonicalAnnotation(
                    id="ann001",
                    image_id="nonexistent",
                    canonical_label="car",
                    canonical_name="car",
                    geometry_type=GeometryType.BBOX,
                    x=0.0,
                    y=0.0,
                    width=10.0,
                    height=10.0,
                ),
            ),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result["all_references_valid"] is False
        broken = result.get("broken_references", [])
        assert isinstance(broken, list) and len(broken) == 1

    @pytest.mark.asyncio
    async def test_duplicate_ids(self, validator: DatasetValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        ds = CanonicalDataset(
            name="dup_ids",
            images=(ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),),
            annotations=(
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
                    id="ann001",
                    image_id="img001",
                    canonical_label="car",
                    canonical_name="car",
                    geometry_type=GeometryType.BBOX,
                    x=10.0,
                    y=10.0,
                    width=10.0,
                    height=10.0,
                ),
            ),
            image_count=1,
            annotation_count=2,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result["duplicate_annotation_ids"] is True

    @pytest.mark.asyncio
    async def test_counts(
        self, validator: DatasetValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset)
        assert result["image_count"] == sample_dataset.image_count
        assert result["annotation_count"] == sample_dataset.annotation_count
        assert result["class_count"] == sample_dataset.class_count

    @pytest.mark.asyncio
    async def test_no_duplicate_annotation_ids(self, validator: DatasetValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        ds = CanonicalDataset(
            name="no_dup",
            images=(ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),),
            annotations=(
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
                    x=10.0,
                    y=10.0,
                    width=10.0,
                    height=10.0,
                ),
            ),
            image_count=1,
            annotation_count=2,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result["duplicate_annotation_ids"] is False

    @pytest.mark.asyncio
    async def test_no_duplicate_image_ids(self, validator: DatasetValidator) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = CanonicalDataset(
            name="no_dup_img",
            images=(
                ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),
                ImageInfo(id="img002", file_path="b.jpg", width=100, height=100),
            ),
            annotations=(),
            image_count=2,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result["duplicate_image_ids"] is False

    @pytest.mark.asyncio
    async def test_duplicate_image_ids_detected(self, validator: DatasetValidator) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = CanonicalDataset(
            name="dup_img_ids",
            images=(
                ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),
                ImageInfo(id="img001", file_path="b.jpg", width=100, height=100),
            ),
            annotations=(),
            image_count=2,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result["duplicate_image_ids"] is True

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import DatasetValidatorInterface

        assert issubclass(DatasetValidator, DatasetValidatorInterface)
