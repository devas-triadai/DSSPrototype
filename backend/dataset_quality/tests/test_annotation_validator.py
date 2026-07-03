from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.dataset_conversion.models import CanonicalAnnotation, CanonicalDataset, ImageInfo
from backend.dataset_quality.annotation_validator import AnnotationValidator


class TestAnnotationValidator:
    @pytest.fixture
    def validator(self) -> AnnotationValidator:
        return AnnotationValidator()

    def _make_dataset(
        self, images: Sequence[ImageInfo], annotations: Sequence[CanonicalAnnotation]
    ) -> CanonicalDataset:
        return CanonicalDataset(
            name="test",
            images=tuple(images),
            annotations=tuple(annotations),
            image_count=len(images),
            annotation_count=len(annotations),
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_valid_dataset(
        self, validator: AnnotationValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_empty_dataset(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = self._make_dataset(
            [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)],
            [],
        )
        result = await validator.validate(ds)
        assert result.missing_annotation_count == 1
        assert "img001" in result.images_without_annotations

    @pytest.mark.asyncio
    async def test_negative_coordinates(
        self, validator: AnnotationValidator, sample_dataset: CanonicalDataset
    ) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType

        anns = list(sample_dataset.annotations) + [
            CanonicalAnnotation(
                id="neg",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=-5.0,
                y=-10.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = list(sample_dataset.images)
        ds = self._make_dataset(imgs, anns)
        result = await validator.validate(ds)
        assert result.negative_coordinate_count == 1
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_zero_area(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="zero",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=10.0,
                y=10.0,
                width=0.0,
                height=0.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = self._make_dataset(imgs, anns)
        result = await validator.validate(ds)
        assert result.zero_area_count == 1
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_out_of_bounds(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="oob",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=600.0,
                y=450.0,
                width=100.0,
                height=100.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = self._make_dataset(imgs, anns)
        result = await validator.validate(ds)
        assert result.out_of_bounds_count == 1

    @pytest.mark.asyncio
    async def test_multiple_issues(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="neg",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=-1.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
            CanonicalAnnotation(
                id="zero",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=0.0,
                height=0.0,
            ),
            CanonicalAnnotation(
                id="oob",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=700.0,
                y=500.0,
                width=100.0,
                height=100.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = self._make_dataset(imgs, anns)
        result = await validator.validate(ds)
        assert result.negative_coordinate_count == 1
        assert result.zero_area_count == 1
        assert result.out_of_bounds_count == 1

    @pytest.mark.asyncio
    async def test_all_images_without_annotations(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [
            ImageInfo(id=f"img{i:03d}", file_path=f"{i}.jpg", width=100, height=100)
            for i in range(5)
        ]
        ds = self._make_dataset(imgs, [])
        result = await validator.validate(ds)
        assert result.missing_annotation_count == 5
        assert len(result.images_without_annotations) == 5

    @pytest.mark.asyncio
    async def test_negative_width_not_area(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation.model_construct(
                id="neg_w",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=10.0,
                y=10.0,
                width=-5.0,
                height=10.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="test",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result.zero_area_count >= 1

    @pytest.mark.asyncio
    async def test_broken_polygon_not_zero_area(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="poly",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.POLYGON,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="test",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result.zero_area_count == 0

    @pytest.mark.asyncio
    async def test_multiple_images_partial_annotations(
        self, validator: AnnotationValidator
    ) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        imgs = [
            ImageInfo(id="img001", file_path="a.jpg", width=640, height=480),
            ImageInfo(id="img002", file_path="b.jpg", width=640, height=480),
            ImageInfo(id="img003", file_path="c.jpg", width=640, height=480),
        ]
        anns = [
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
        ]
        ds = CanonicalDataset(
            name="partial",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=3,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result.missing_annotation_count == 2
        assert "img002" in result.images_without_annotations
        assert "img003" in result.images_without_annotations

    @pytest.mark.asyncio
    async def test_no_image_map_available(self, validator: AnnotationValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
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
        ]
        imgs = [ImageInfo(id="img001", file_path="a.jpg", width=None, height=None)]
        ds = CanonicalDataset(
            name="no_dims",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds)
        assert result.out_of_bounds_count == 0

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import AnnotationValidatorInterface

        assert issubclass(AnnotationValidator, AnnotationValidatorInterface)
