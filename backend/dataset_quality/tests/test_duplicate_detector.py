from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.duplicate_detector import DuplicateDetector


class TestDuplicateDetector:
    @pytest.fixture
    def detector(self) -> DuplicateDetector:
        return DuplicateDetector()

    @pytest.mark.asyncio
    async def test_no_duplicates(
        self, detector: DuplicateDetector, sample_dataset: CanonicalDataset
    ) -> None:
        result = await detector.detect(sample_dataset)
        assert result.total_duplicate_images == 0
        assert result.total_duplicate_annotations == 0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_duplicate_annotations_detected(
        self, detector: DuplicateDetector, sample_duplicate_dataset: CanonicalDataset
    ) -> None:
        result = await detector.detect(sample_duplicate_dataset)
        assert result.total_duplicate_annotations >= 1

    @pytest.mark.asyncio
    async def test_repeated_ids_detected(self, detector: DuplicateDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        ds = CanonicalDataset(
            name="repeated",
            images=(ImageInfo(id="dup_id", file_path="a.jpg", width=100, height=100),),
            annotations=(
                CanonicalAnnotation(
                    id="dup_id",
                    image_id="dup_id",
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
                    image_id="dup_id",
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
        result = await detector.detect(ds)
        assert len(result.repeated_ids) >= 1
        assert not result.passed

    @pytest.mark.asyncio
    async def test_near_duplicate_images(self, detector: DuplicateDetector) -> None:
        from backend.dataset_conversion.models import ImageInfo

        same_spec = ImageInfo(
            id="img001",
            file_path="/data/a.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
        )
        same_spec2 = ImageInfo(
            id="img002",
            file_path="/data/b.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="rgb",
        )
        ds = CanonicalDataset(
            name="near_dup",
            images=(same_spec, same_spec2),
            annotations=(),
            image_count=2,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.near_duplicate_image_pairs) >= 1

    @pytest.mark.asyncio
    async def test_duplicate_image_hash(self, detector: DuplicateDetector) -> None:
        from backend.dataset_conversion.models import ImageInfo

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"same content")
            path1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"same content")
            path2 = f.name
        try:
            img1 = ImageInfo(
                id="img001", file_path=path1, width=100, height=100, format="jpg", color_space="rgb"
            )
            img2 = ImageInfo(
                id="img002", file_path=path2, width=100, height=100, format="jpg", color_space="rgb"
            )
            ds = CanonicalDataset(
                name="hash_dup",
                images=(img1, img2),
                annotations=(),
                image_count=2,
                annotation_count=0,
                class_count=0,
                ontology_version="1.0.0",
                pipeline_version="1.0.0",
            )
            result = await detector.detect(ds)
            assert result.total_duplicate_images >= 1
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, detector: DuplicateDetector, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await detector.detect(sample_empty_dataset)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_duplicate_annotations(self, detector: DuplicateDetector) -> None:
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
                canonical_label="person",
                canonical_name="person",
                geometry_type=GeometryType.BBOX,
                x=20.0,
                y=20.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="/data/a.jpg", width=100, height=100)]
        ds = CanonicalDataset(
            name="no_dup_anns",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=2,
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert result.total_duplicate_annotations == 0

    @pytest.mark.asyncio
    async def test_duplicate_annotations_same_coords(self, detector: DuplicateDetector) -> None:
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
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="/data/a.jpg", width=100, height=100)]
        ds = CanonicalDataset(
            name="dup_anns",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=2,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert result.total_duplicate_annotations >= 1

    @pytest.mark.asyncio
    async def test_no_repeated_ids(self, detector: DuplicateDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        imgs = [ImageInfo(id="img001", file_path="/data/a.jpg", width=100, height=100)]
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
        ]
        ds = CanonicalDataset(
            name="no_dup_ids",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.repeated_ids) == 0

    @pytest.mark.asyncio
    async def test_near_duplicate_different_format(self, detector: DuplicateDetector) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [
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
                file_path="/data/b.png",
                width=640,
                height=480,
                format="png",
                color_space="rgb",
            ),
        ]
        ds = CanonicalDataset(
            name="near_dup_diff_fmt",
            images=tuple(imgs),
            annotations=(),
            image_count=2,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.near_duplicate_image_pairs) == 0

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import DuplicateDetectorInterface

        assert issubclass(DuplicateDetector, DuplicateDetectorInterface)
