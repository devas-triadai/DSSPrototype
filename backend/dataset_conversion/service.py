from __future__ import annotations

from backend.dataset_conversion.annotation_converter import AnnotationConverter
from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.conversion_pipeline import ConversionPipeline
from backend.dataset_conversion.dataset_exporter import DatasetExporter
from backend.dataset_conversion.dataset_loader import DatasetLoader
from backend.dataset_conversion.dataset_merger import DatasetMerger
from backend.dataset_conversion.dataset_splitter import DatasetSplitter
from backend.dataset_conversion.dataset_statistics import DatasetStatisticsGenerator
from backend.dataset_conversion.dataset_validator import DatasetValidator
from backend.dataset_conversion.image_converter import ImageConverter
from backend.dataset_conversion.interfaces import DatasetConversionServiceInterface
from backend.dataset_conversion.manifest_builder import ManifestBuilder
from backend.dataset_conversion.models import (
    CanonicalDataset,
    ConvertResult,
    DatasetManifest,
    DatasetStatistics,
    ExportResult,
    LoadResult,
    MergeConfig,
    MergeResult,
    SplitConfig,
    SplitResult,
    ValidationReport,
)
from backend.dataset_conversion.ontology_adapter import OntologyAdapter


class DatasetConversionService(DatasetConversionServiceInterface):
    def __init__(
        self,
        loader: DatasetLoader | None = None,
        ontology_adapter: OntologyAdapter | None = None,
        annotation_converter: AnnotationConverter | None = None,
        image_converter: ImageConverter | None = None,
        merger: DatasetMerger | None = None,
        splitter: DatasetSplitter | None = None,
        validator: DatasetValidator | None = None,
        statistics: DatasetStatisticsGenerator | None = None,
        manifest_builder: ManifestBuilder | None = None,
        exporter: DatasetExporter | None = None,
        pipeline: ConversionPipeline | None = None,
        config: DatasetConversionConfig | None = None,
    ) -> None:
        cfg = config or dataset_conversion_config
        self._ontology_adapter = ontology_adapter or OntologyAdapter(config=cfg)
        self._loader = loader or DatasetLoader(config=cfg)
        self._annotation_converter = annotation_converter or AnnotationConverter(
            ontology_adapter=self._ontology_adapter,
        )
        self._image_converter = image_converter or ImageConverter(config=cfg)
        self._merger = merger or DatasetMerger()
        self._splitter = splitter or DatasetSplitter()
        self._validator = validator or DatasetValidator(
            ontology_adapter=self._ontology_adapter,
        )
        self._statistics = statistics or DatasetStatisticsGenerator()
        self._manifest_builder = manifest_builder or ManifestBuilder()
        self._exporter = exporter or DatasetExporter(config=cfg)
        self._pipeline = pipeline or ConversionPipeline(
            loader=self._loader,
            ontology_adapter=self._ontology_adapter,
            annotation_converter=self._annotation_converter,
            image_converter=self._image_converter,
            validator=self._validator,
            statistics=self._statistics,
            manifest_builder=self._manifest_builder,
            config=cfg,
        )

    async def load_dataset(
        self,
        path: str,
        dataset_format: str,
        **kwargs: str,
    ) -> LoadResult:
        return await self._loader.load(path, dataset_format, **kwargs)

    async def convert_dataset(
        self,
        load_result: LoadResult,
        dataset_name: str,
    ) -> CanonicalDataset:
        result = await self._pipeline._convert(load_result, dataset_name)
        return result

    async def run_pipeline(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        output_path: str,
        **kwargs: str,
    ) -> ConvertResult:
        return await self._pipeline.run(
            source_path=source_path,
            source_format=source_format,
            dataset_name=dataset_name,
            output_path=output_path,
            **kwargs,
        )

    async def merge_datasets(
        self,
        datasets: list[CanonicalDataset],
        config: MergeConfig | None = None,
    ) -> MergeResult:
        return await self._merger.merge(datasets, config)

    async def split_dataset(
        self,
        dataset: CanonicalDataset,
        config: SplitConfig | None = None,
    ) -> SplitResult:
        return await self._splitter.split(dataset, config)

    async def validate_dataset(
        self,
        dataset: CanonicalDataset,
    ) -> ValidationReport:
        return await self._validator.validate(dataset)

    async def dataset_statistics(
        self,
        dataset: CanonicalDataset,
    ) -> DatasetStatistics:
        return await self._statistics.compute(dataset)

    async def export_dataset(
        self,
        dataset: CanonicalDataset,
        export_format: str,
        output_path: str,
    ) -> ExportResult:
        return await self._exporter.export(dataset, export_format, output_path)

    async def build_manifest(
        self,
        dataset: CanonicalDataset,
        statistics: DatasetStatistics | None = None,
    ) -> DatasetManifest:
        return await self._manifest_builder.build(dataset, statistics)
