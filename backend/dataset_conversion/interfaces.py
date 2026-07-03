from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ConvertResult,
    DatasetManifest,
    DatasetStatistics,
    ExportResult,
    ImageInfo,
    LoadResult,
    MergeConfig,
    MergeResult,
    SourceAnnotation,
    SourceCategory,
    SplitConfig,
    SplitResult,
    ValidationReport,
)


class DatasetLoaderInterface(ABC):
    @abstractmethod
    async def load(
        self,
        path: str,
        dataset_format: str,
        **kwargs: str,
    ) -> LoadResult: ...

    @abstractmethod
    async def load_image(self, image_id: str) -> ImageInfo | None: ...

    @abstractmethod
    async def supported_formats(self) -> list[str]: ...


class AnnotationLoaderInterface(ABC):
    @abstractmethod
    async def parse_annotations(
        self,
        data: str,
        dataset_format: str,
    ) -> list[SourceAnnotation]: ...

    @abstractmethod
    async def parse_categories(
        self,
        data: str,
        dataset_format: str,
    ) -> list[SourceCategory]: ...


class OntologyAdapterInterface(ABC):
    @abstractmethod
    async def translate_label(
        self,
        source_label: str,
        dataset_name: str,
    ) -> str: ...

    @abstractmethod
    async def translate_batch(
        self,
        source_labels: list[str],
        dataset_name: str,
    ) -> dict[str, str]: ...

    @abstractmethod
    async def is_known_label(self, canonical_label: str) -> bool: ...


class GeometryConverterInterface(ABC):
    @abstractmethod
    async def to_canonical_bbox(
        self,
        geometry: Sequence[float],
        source_format: str,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> tuple[float, float, float, float]: ...

    @abstractmethod
    async def normalize(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]: ...

    @abstractmethod
    async def denormalize(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]: ...


class AnnotationConverterInterface(ABC):
    @abstractmethod
    async def convert_annotation(
        self,
        source: SourceAnnotation,
        canonical_label: str,
    ) -> CanonicalAnnotation: ...

    @abstractmethod
    async def convert_batch(
        self,
        sources: list[SourceAnnotation],
        label_map: dict[str, str],
    ) -> list[CanonicalAnnotation]: ...


class ImageConverterInterface(ABC):
    @abstractmethod
    async def validate_image(self, image: ImageInfo) -> list[str]: ...

    @abstractmethod
    async def standardize_metadata(self, image: ImageInfo) -> ImageInfo: ...


class DatasetMergerInterface(ABC):
    @abstractmethod
    async def merge(
        self,
        datasets: list[CanonicalDataset],
        config: MergeConfig | None = None,
    ) -> MergeResult: ...


class DatasetSplitterInterface(ABC):
    @abstractmethod
    async def split(
        self,
        dataset: CanonicalDataset,
        config: SplitConfig | None = None,
    ) -> SplitResult: ...


class DatasetValidatorInterface(ABC):
    @abstractmethod
    async def validate(
        self,
        dataset: CanonicalDataset,
    ) -> ValidationReport: ...


class DatasetStatisticsInterface(ABC):
    @abstractmethod
    async def compute(
        self,
        dataset: CanonicalDataset,
    ) -> DatasetStatistics: ...


class ManifestBuilderInterface(ABC):
    @abstractmethod
    async def build(
        self,
        dataset: CanonicalDataset,
        statistics: DatasetStatistics | None = None,
    ) -> DatasetManifest: ...


class DatasetExporterInterface(ABC):
    @abstractmethod
    async def export(
        self,
        dataset: CanonicalDataset,
        export_format: str,
        output_path: str,
    ) -> ExportResult: ...

    @abstractmethod
    async def supported_export_formats(self) -> list[str]: ...


class ConversionPipelineInterface(ABC):
    @abstractmethod
    async def run(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        output_path: str,
        **kwargs: str,
    ) -> ConvertResult: ...


class DatasetConversionServiceInterface(ABC):
    @abstractmethod
    async def load_dataset(
        self,
        path: str,
        dataset_format: str,
        **kwargs: str,
    ) -> LoadResult: ...

    @abstractmethod
    async def convert_dataset(
        self,
        load_result: LoadResult,
        dataset_name: str,
    ) -> CanonicalDataset: ...

    @abstractmethod
    async def merge_datasets(
        self,
        datasets: list[CanonicalDataset],
        config: MergeConfig | None = None,
    ) -> MergeResult: ...

    @abstractmethod
    async def split_dataset(
        self,
        dataset: CanonicalDataset,
        config: SplitConfig | None = None,
    ) -> SplitResult: ...

    @abstractmethod
    async def validate_dataset(
        self,
        dataset: CanonicalDataset,
    ) -> ValidationReport: ...

    @abstractmethod
    async def dataset_statistics(
        self,
        dataset: CanonicalDataset,
    ) -> DatasetStatistics: ...

    @abstractmethod
    async def export_dataset(
        self,
        dataset: CanonicalDataset,
        export_format: str,
        output_path: str,
    ) -> ExportResult: ...

    @abstractmethod
    async def build_manifest(
        self,
        dataset: CanonicalDataset,
        statistics: DatasetStatistics | None = None,
    ) -> DatasetManifest: ...
