from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.consistency_checker import ConsistencyChecker


class TestConsistencyChecker:
    @pytest.fixture
    def checker(self) -> ConsistencyChecker:
        return ConsistencyChecker()

    @pytest.mark.asyncio
    async def test_consistent_dataset(
        self, checker: ConsistencyChecker, sample_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_dataset)
        assert result.passed is True
        assert result.metadata_consistent is True
        assert result.ontology_consistent is True
        assert result.version_consistent is True

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, checker: ConsistencyChecker, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_empty_dataset)
        assert result.ontology_consistent is False
        assert not result.passed

    @pytest.mark.asyncio
    async def test_missing_version(self, checker: ConsistencyChecker) -> None:

        ds = CanonicalDataset.model_construct(
            name="no_ver",
            images=(),
            annotations=(),
            image_count=0,
            annotation_count=0,
            class_count=0,
            ontology_version="",
            pipeline_version="",
        )
        result = await checker.check(ds)
        assert result.version_consistent is False

    @pytest.mark.asyncio
    async def test_annotation_consistency(self, checker: ConsistencyChecker) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="a1",
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
                id="a2",
                image_id="img002",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = [
            ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),
            ImageInfo(id="img002", file_path="b.jpg", width=100, height=100),
        ]
        ds = CanonicalDataset(
            name="consistent_anns",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=2,
            annotation_count=2,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert result.annotation_consistent is True

    @pytest.mark.asyncio
    async def test_all_consistent_with_data(self, checker: ConsistencyChecker) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="a1",
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
                id="a2",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=20.0,
                y=20.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="a.jpg", width=100, height=100)]
        ds = CanonicalDataset.model_construct(
            name="consistent",
            images=(imgs[0],),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=2,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert result.metadata_consistent is True
        assert result.split_consistent is True
        assert result.annotation_consistent is True
        assert result.version_consistent is True

    @pytest.mark.asyncio
    async def test_no_annotations_but_valid(self, checker: ConsistencyChecker) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = CanonicalDataset(
            name="no_anns",
            images=(ImageInfo(id="img001", file_path="a.jpg", width=100, height=100),),
            annotations=(),
            image_count=1,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert result.ontology_consistent is False

    @pytest.mark.asyncio
    async def test_metadata_inconsistent(self, checker: ConsistencyChecker) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [
            ImageInfo(
                id="img001",
                file_path="a.jpg",
                width=100,
                height=100,
                metadata={"pipeline_version": "1.0.0"},
            ),
            ImageInfo(
                id="img002",
                file_path="b.jpg",
                width=100,
                height=100,
                metadata={"pipeline_version": "2.0.0"},
            ),
        ]
        ds = CanonicalDataset(
            name="meta_inconsistent",
            images=tuple(imgs),
            annotations=(),
            image_count=2,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert not result.metadata_consistent

    @pytest.mark.asyncio
    async def test_split_label_in_dataset_metadata(self, checker: ConsistencyChecker) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [ImageInfo(id="img001", file_path="a.jpg", width=100, height=100)]
        ds = CanonicalDataset.model_construct(
            name="split_test",
            images=(imgs[0],),
            annotations=(),
            image_count=1,
            annotation_count=0,
            class_count=0,
            metadata={"split": "train"},
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert result.split_consistent is True

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import ConsistencyCheckerInterface

        assert issubclass(ConsistencyChecker, ConsistencyCheckerInterface)
