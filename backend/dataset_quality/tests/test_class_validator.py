from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.class_validator import ClassValidator
from backend.dataset_quality.config import DatasetQualityConfig


class TestClassValidator:
    @pytest.fixture
    def validator(self) -> ClassValidator:
        return ClassValidator()

    @pytest.fixture
    def sensitive_validator(self) -> ClassValidator:
        cfg = DatasetQualityConfig(rare_class_threshold=0.2, min_class_samples=3)
        return ClassValidator(cfg)

    @pytest.mark.asyncio
    async def test_valid_dataset(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(
            sample_dataset, ["ground_vehicle.car", "ground_vehicle.truck", "people.person"]
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unknown_class_detected(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset, ["people.person"])
        assert result.unknown_class_count >= 1
        assert not result.passed

    @pytest.mark.asyncio
    async def test_unknown_classes_count(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset, ["aerial.drone"])
        assert result.unknown_class_count >= 1

    @pytest.mark.asyncio
    async def test_unused_classes_detected(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        all_classes = [
            "ground_vehicle.car",
            "ground_vehicle.truck",
            "people.person",
            "aerial.drone",
        ]
        result = await validator.validate(sample_dataset, all_classes)
        assert result.unused_class_count >= 1

    @pytest.mark.asyncio
    async def test_no_ontology_provided(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset)
        assert result.passed is True
        assert result.unknown_class_count == 0
        assert result.unused_class_count == 0

    @pytest.mark.asyncio
    async def test_rare_class_detected(
        self, sensitive_validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await sensitive_validator.validate(
            sample_dataset, ["ground_vehicle.car", "ground_vehicle.truck", "people.person"]
        )
        assert result.rare_class_count >= 1

    @pytest.mark.asyncio
    async def test_imbalance_ratio(self, validator: ClassValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id=f"ann{i:03d}",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(100)
        ] + [
            CanonicalAnnotation(
                id=f"ann{i:03d}",
                image_id="img001",
                canonical_label="person",
                canonical_name="person",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(1, 3)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="imb",
            images=(imgs[0],),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds, ["car", "person"])
        assert result.imbalance_ratio >= 50.0

    @pytest.mark.asyncio
    async def test_class_distribution(
        self, validator: ClassValidator, sample_dataset: CanonicalDataset
    ) -> None:
        result = await validator.validate(sample_dataset)
        assert "ground_vehicle.car" in result.class_distribution
        assert "people.person" in result.class_distribution

    @pytest.mark.asyncio
    async def test_empty_annotation_dataset(self, validator: ClassValidator) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = CanonicalDataset(
            name="empty",
            images=(ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),),
            annotations=(),
            image_count=1,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds, ["car", "person"])
        assert result.total_classes == 0

    @pytest.mark.asyncio
    async def test_all_classes_unknown(self, validator: ClassValidator) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="ann001",
                image_id="img001",
                canonical_label="unknown_cls",
                canonical_name="unknown",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
        ]
        imgs = [ImageInfo(id="img001", file_path="a.jpg", width=100, height=100)]
        ds = CanonicalDataset(
            name="all_unknown",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds, ["car", "person"])
        assert result.unknown_class_count == 1
        assert not result.passed

    @pytest.mark.asyncio
    async def test_balance_ratio_single_class(self, validator: ClassValidator) -> None:
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
            )
            for _ in range(10)
        ]
        imgs = [ImageInfo(id="img001", file_path="a.jpg", width=100, height=100)]
        ds = CanonicalDataset(
            name="single",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await validator.validate(ds, ["car"])
        assert result.imbalance_ratio == 1.0

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import ClassValidatorInterface

        assert issubclass(ClassValidator, ClassValidatorInterface)
