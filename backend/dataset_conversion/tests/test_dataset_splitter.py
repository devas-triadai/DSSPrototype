from __future__ import annotations

import pytest

from backend.dataset_conversion.dataset_splitter import DatasetSplitter
from backend.dataset_conversion.models import (
    CanonicalDataset,
    SplitConfig,
    SplitStrategy,
)


class TestDatasetSplitter:
    @pytest.fixture
    def splitter(self) -> DatasetSplitter:
        return DatasetSplitter()

    @pytest.mark.asyncio
    async def test_random_split(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
        sample_split_config: SplitConfig,
    ) -> None:
        result = await splitter.split(sample_canonical_dataset, sample_split_config)
        assert result.train.image_count > 0
        assert result.val.image_count > 0
        assert result.test.image_count >= 0
        total = result.train.image_count + result.val.image_count + result.test.image_count
        assert total == sample_canonical_dataset.image_count

    @pytest.mark.asyncio
    async def test_random_split_default_config(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = await splitter.split(sample_canonical_dataset)
        assert result.train.image_count > 0
        assert 0.5 < result.train_ratio < 0.85

    @pytest.mark.asyncio
    async def test_stratified_split(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        result = await splitter.split(sample_canonical_dataset, config)
        total = result.train.image_count + result.val.image_count + result.test.image_count
        assert total == sample_canonical_dataset.image_count

    @pytest.mark.asyncio
    async def test_class_balanced_split(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(
            strategy=SplitStrategy.CLASS_BALANCED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        result = await splitter.split(sample_canonical_dataset, config)
        assert result.train.image_count > 0

    @pytest.mark.asyncio
    async def test_split_deterministic(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
        sample_split_config: SplitConfig,
    ) -> None:
        result1 = await splitter.split(sample_canonical_dataset, sample_split_config)
        result2 = await splitter.split(sample_canonical_dataset, sample_split_config)
        assert result1.train.image_count == result2.train.image_count
        assert result1.val.image_count == result2.val.image_count

    @pytest.mark.asyncio
    async def test_split_ratios_sum_to_one(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        result = await splitter.split(sample_canonical_dataset, config)
        assert abs(result.train_ratio + result.val_ratio + result.test_ratio - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_split_preserves_annotations(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = await splitter.split(sample_canonical_dataset)
        total_anns = (
            result.train.annotation_count
            + result.val.annotation_count
            + result.test.annotation_count
        )
        assert total_anns == sample_canonical_dataset.annotation_count

    @pytest.mark.asyncio
    async def test_split_unknown_strategy(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig.model_construct(
            strategy="unknown",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        with pytest.raises(ValueError, match="Unknown split strategy"):
            await splitter.split(sample_canonical_dataset, config)

    @pytest.mark.asyncio
    async def test_split_no_shuffle(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, shuffle=False
        )
        result = await splitter.split(sample_canonical_dataset, config)
        assert result.train.image_count > 0

    @pytest.mark.asyncio
    async def test_split_validation_error_on_bad_ratios(
        self,
        splitter: DatasetSplitter,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(train_ratio=0.9, val_ratio=0.2, test_ratio=0.2, seed=42)
        with pytest.raises(ValueError, match="Split ratios sum to"):
            await splitter.split(sample_canonical_dataset, config)
