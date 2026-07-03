from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.integrity_checker import IntegrityChecker


class TestIntegrityChecker:
    @pytest.fixture
    def checker(self) -> IntegrityChecker:
        return IntegrityChecker()

    @pytest.mark.asyncio
    async def test_missing_files(
        self, checker: IntegrityChecker, sample_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_dataset)
        assert not result.all_files_present
        assert len(result.missing_files) == sample_dataset.image_count
        assert not result.passed

    @pytest.mark.asyncio
    async def test_all_files_present(self, checker: IntegrityChecker) -> None:
        from backend.dataset_conversion.models import ImageInfo

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"test data")
            path = f.name
        try:
            img = ImageInfo(
                id="img001", file_path=path, width=100, height=100, format="jpg", color_space="rgb"
            )
            ds = CanonicalDataset(
                name="present",
                images=(img,),
                annotations=(),
                image_count=1,
                annotation_count=0,
                class_count=0,
                ontology_version="1.0.0",
                pipeline_version="1.0.0",
            )
            result = await checker.check(ds)
            assert result.all_files_present is True
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_broken_references(self, checker: IntegrityChecker) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
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
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=100, height=100)]
        ds = CanonicalDataset(
            name="broken_refs",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await checker.check(ds)
        assert not result.no_broken_references
        assert len(result.broken_references) >= 1

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, checker: IntegrityChecker, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_empty_dataset)
        assert result.all_files_present is True
        assert result.no_broken_references is True

    @pytest.mark.asyncio
    async def test_missing_files_count(
        self, checker: IntegrityChecker, sample_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_dataset)
        assert len(result.missing_files) == sample_dataset.image_count

    @pytest.mark.asyncio
    async def test_no_broken_references(
        self, checker: IntegrityChecker, sample_dataset: CanonicalDataset
    ) -> None:
        result = await checker.check(sample_dataset)
        assert result.no_broken_references is True
        assert len(result.broken_references) == 0

    @pytest.mark.asyncio
    async def test_version_info_present(self, checker: IntegrityChecker) -> None:
        from backend.dataset_conversion.models import ImageInfo

        ds = CanonicalDataset.model_construct(
            name="versioned",
            images=(ImageInfo(id="img001", file_path="nonexistent.jpg", width=100, height=100),),
            annotations=(),
            image_count=1,
            annotation_count=0,
            class_count=0,
            ontology_version="",
            pipeline_version="",
        )
        result = await checker.check(ds)
        assert result.version_valid is False

    @pytest.mark.asyncio
    async def test_with_image_dir(self, checker: IntegrityChecker) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"data")
            path = f.name
        import os

        d = os.path.dirname(path)
        fname = os.path.basename(path)
        try:
            from backend.dataset_conversion.models import ImageInfo

            img = ImageInfo(
                id="img001", file_path=fname, width=100, height=100, format="jpg", color_space="rgb"
            )
            ds = CanonicalDataset(
                name="with_dir",
                images=(img,),
                annotations=(),
                image_count=1,
                annotation_count=0,
                class_count=0,
                ontology_version="1.0.0",
                pipeline_version="1.0.0",
            )
            result = await checker.check(ds, image_dir=d)
            assert result.all_files_present is True
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import IntegrityCheckerInterface

        assert issubclass(IntegrityChecker, IntegrityCheckerInterface)
