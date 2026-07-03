from __future__ import annotations

from backend.dataset_conversion.config import DatasetConversionConfig


class TestDatasetConversionConfig:
    def test_defaults(self) -> None:
        config = DatasetConversionConfig()
        assert config.version == "1.0.0"
        assert config.strict_mode is True
        assert config.default_train_ratio == 0.7
        assert config.default_seed == 42
        assert config.target_image_format == "png"

    def test_env_prefix(self) -> None:
        assert DatasetConversionConfig.model_config["env_prefix"] == "DATASET_CONVERSION_"

    def test_custom_values(self) -> None:
        config = DatasetConversionConfig(
            version="2.0.0",
            strict_mode=False,
            default_seed=123,
        )
        assert config.version == "2.0.0"
        assert config.strict_mode is False
        assert config.default_seed == 123


class TestExceptions:
    from backend.dataset_conversion.exceptions import (
        AnnotationError,
        ConversionError,
        ExportError,
        GeometryError,
        ImageError,
        LoadError,
        ManifestError,
        MergeError,
        OntologyAdapterError,
        PipelineError,
        SplitError,
        StatisticsError,
        ValidationError,
    )

    def test_all_inherit_from_conversion_error(self) -> None:
        for exc_class in [
            self.AnnotationError,
            self.ExportError,
            self.GeometryError,
            self.ImageError,
            self.LoadError,
            self.ManifestError,
            self.MergeError,
            self.OntologyAdapterError,
            self.PipelineError,
            self.SplitError,
            self.StatisticsError,
            self.ValidationError,
        ]:
            assert issubclass(exc_class, self.ConversionError)

    def test_exception_instantiation(self) -> None:
        for exc_class in [
            self.AnnotationError,
            self.ExportError,
            self.GeometryError,
        ]:
            exc = exc_class("test error")
            assert str(exc) == "test error"


class TestInterfaces:
    from backend.dataset_conversion.interfaces import (
        AnnotationConverterInterface,
        AnnotationLoaderInterface,
        ConversionPipelineInterface,
        DatasetConversionServiceInterface,
        DatasetExporterInterface,
        DatasetLoaderInterface,
        DatasetMergerInterface,
        DatasetSplitterInterface,
        DatasetStatisticsInterface,
        DatasetValidatorInterface,
        GeometryConverterInterface,
        ImageConverterInterface,
        ManifestBuilderInterface,
        OntologyAdapterInterface,
    )

    def test_all_interfaces_defined(self) -> None:
        assert hasattr(self.AnnotationConverterInterface, "convert_annotation")
        assert hasattr(self.DatasetLoaderInterface, "load")
        assert hasattr(self.DatasetExporterInterface, "export")
        assert hasattr(self.DatasetMergerInterface, "merge")
        assert hasattr(self.DatasetSplitterInterface, "split")
        assert hasattr(self.DatasetValidatorInterface, "validate")
        assert hasattr(self.ManifestBuilderInterface, "build")
        assert hasattr(self.OntologyAdapterInterface, "translate_label")
        assert hasattr(self.GeometryConverterInterface, "to_canonical_bbox")
        assert hasattr(self.ConversionPipelineInterface, "run")
        assert hasattr(self.DatasetConversionServiceInterface, "load_dataset")
        assert hasattr(self.DatasetConversionServiceInterface, "convert_dataset")
        assert hasattr(self.DatasetConversionServiceInterface, "merge_datasets")
