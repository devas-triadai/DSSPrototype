from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.outlier_detector import OutlierDetector


class TestOutlierDetector:
    @pytest.fixture
    def detector(self) -> OutlierDetector:
        return OutlierDetector()

    @pytest.mark.asyncio
    async def test_no_outliers(
        self, detector: OutlierDetector, sample_dataset: CanonicalDataset
    ) -> None:
        result = await detector.detect(sample_dataset)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_tiny_objects(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="tiny",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=1.0,
                height=1.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="tiny_obj",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.tiny_objects) >= 1

    @pytest.mark.asyncio
    async def test_extreme_aspect_ratio(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="extreme",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=1.0,
                height=1000.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="extreme_ar",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.extreme_aspect_ratios) >= 1

    @pytest.mark.asyncio
    async def test_huge_objects(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="huge",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10000.0,
                height=10000.0,
            ),
        ]
        normal_anns = [
            CanonicalAnnotation(
                id=f"normal_{i}",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            )
            for i in range(10)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=20000, height=20000)]
        ds = CanonicalDataset(
            name="huge_obj",
            images=tuple(imgs),
            annotations=tuple(anns + normal_anns),
            image_count=1,
            annotation_count=len(anns + normal_anns),
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.huge_objects) >= 1

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, detector: OutlierDetector, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await detector.detect(sample_empty_dataset)
        assert result.passed is True
        assert result.total_outliers == 0

    @pytest.mark.asyncio
    async def test_no_tiny_objects(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="normal",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=100.0,
                height=100.0,
            )
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="no_tiny",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.tiny_objects) == 0

    @pytest.mark.asyncio
    async def test_no_extreme_aspect_ratios(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="normal",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=100.0,
                height=50.0,
            )
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="no_extreme",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.extreme_aspect_ratios) == 0

    @pytest.mark.asyncio
    async def test_abnormal_resolution(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [
            ImageInfo(id=f"normal{i}", file_path=f"{i}.jpg", width=640, height=480)
            for i in range(20)
        ] + [
            ImageInfo(id="weird", file_path="weird.jpg", width=32000, height=24000),
        ]
        ds = CanonicalDataset(
            name="abnormal_res",
            images=tuple(imgs),
            annotations=(),
            image_count=len(imgs),
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert len(result.abnormal_resolutions) >= 1

    @pytest.mark.asyncio
    async def test_outlier_counts(self, detector: OutlierDetector) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="tiny",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=1.0,
                height=1.0,
            )
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="outlier_ct",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await detector.detect(ds)
        assert result.total_outliers >= 1

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import OutlierDetectorInterface

        assert issubclass(OutlierDetector, OutlierDetectorInterface)
