from __future__ import annotations

from backend.dataset_conversion.annotation_converter import AnnotationConverter
from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.dataset_loader import DatasetLoader
from backend.dataset_conversion.dataset_statistics import DatasetStatisticsGenerator
from backend.dataset_conversion.dataset_validator import DatasetValidator
from backend.dataset_conversion.exceptions import PipelineError
from backend.dataset_conversion.image_converter import ImageConverter
from backend.dataset_conversion.manifest_builder import ManifestBuilder
from backend.dataset_conversion.models import (
    CanonicalDataset,
    ConversionReport,
    ConvertResult,
    ImageInfo,
    LoadResult,
)
from backend.dataset_conversion.ontology_adapter import OntologyAdapter


class ConversionPipeline:
    def __init__(
        self,
        loader: DatasetLoader | None = None,
        ontology_adapter: OntologyAdapter | None = None,
        annotation_converter: AnnotationConverter | None = None,
        image_converter: ImageConverter | None = None,
        validator: DatasetValidator | None = None,
        statistics: DatasetStatisticsGenerator | None = None,
        manifest_builder: ManifestBuilder | None = None,
        config: DatasetConversionConfig | None = None,
    ) -> None:
        self._loader = loader or DatasetLoader(config=config)
        self._ontology_adapter = ontology_adapter or OntologyAdapter(config=config)
        self._annotation_converter = annotation_converter or AnnotationConverter(
            ontology_adapter=self._ontology_adapter,
        )
        self._image_converter = image_converter or ImageConverter(config=config)
        self._validator = validator or DatasetValidator(
            ontology_adapter=self._ontology_adapter,
        )
        self._statistics = statistics or DatasetStatisticsGenerator()
        self._manifest_builder = manifest_builder or ManifestBuilder()
        self._config = config or dataset_conversion_config

    async def run(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        output_path: str,
        **kwargs: str,
    ) -> ConvertResult:
        try:
            load_kwargs = dict(kwargs)
            load_kwargs.setdefault("data_dir", "")
            load_kwargs["dataset_name"] = dataset_name
            load_result = await self._loader.load(
                path=source_path,
                dataset_format=source_format,
                **load_kwargs,
            )
        except Exception as e:
            raise PipelineError(f"Failed to load dataset: {e}") from e

        canonical_dataset = await self._convert(load_result, dataset_name)

        validation = await self._validator.validate(canonical_dataset)
        if not validation.valid and self._config.strict_mode:
            raise PipelineError(f"Validation failed: {'; '.join(validation.errors)}")

        return ConvertResult(
            dataset=canonical_dataset,
            report=ConversionReport(
                total_images=load_result.image_count,
                total_annotations=load_result.annotation_count,
                mapped_annotations=canonical_dataset.annotation_count,
                unmapped_annotations=load_result.annotation_count
                - canonical_dataset.annotation_count,
                skipped_images=0,
                errors=validation.errors if not validation.valid else tuple(),
            ),
        )

    async def _convert(
        self,
        load_result: LoadResult,
        dataset_name: str,
    ) -> CanonicalDataset:
        images: list[ImageInfo] = []
        for src_img in load_result.images:
            img = await self._image_converter.standardize_metadata(src_img)
            if self._config.validate_images:
                await self._image_converter.validate_image(img)
            images.append(img)

        unique_labels = set[str]()
        for ann in load_result.annotations:
            unique_labels.add(ann.category_name)

        label_map: dict[str, str] = {}
        for label in sorted(unique_labels):
            canonical = await self._ontology_adapter.translate_label(
                source_label=label,
                dataset_name=dataset_name,
            )
            label_map[label] = canonical

        converted_annotations = await self._annotation_converter.convert_batch(
            sources=list(load_result.annotations),
            label_map=label_map,
        )

        all_labels = {a.canonical_label for a in converted_annotations}

        return CanonicalDataset(
            name=dataset_name,
            images=tuple(images),
            annotations=tuple(converted_annotations),
            image_count=len(images),
            annotation_count=len(converted_annotations),
            class_count=len(all_labels),
            source_datasets=(load_result.dataset_name,),
            pipeline_version=self._config.version,
        )
