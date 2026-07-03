"""Tests for Pydantic data models."""

from datetime import datetime

from backend.dataset_intelligence.models import (
    Annotation,
    DatasetIntelligenceRegistryEntry,
    DuplicateEntry,
    DuplicateReport,
    ExportResult,
    HarmonizedDataset,
    ImageRecord,
    ImportResult,
    MergedDataset,
    NormalizedDataset,
    OntologyMappingReport,
    ProcessedDataset,
    ProvenanceRecord,
    QualityReport,
    RawDataset,
    StatisticsReport,
    ValidationReport,
)


def test_annotation_defaults() -> None:
    ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    assert ann.normalized_class == ""
    assert ann.ontology_class == ""
    assert ann.bbox_format == "xyxy_normalized"
    assert ann.confidence is None
    assert ann.segmentation is None


def test_annotation_is_frozen() -> None:
    ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    try:
        ann.class_name = "changed"
        assert False, "Should be frozen"
    except Exception:
        assert True


def test_image_record_defaults() -> None:
    img = ImageRecord(
        image_id="img01",
        image_path="/path/img.jpg",
        image_name="img.jpg",
    )
    assert img.width == 0
    assert img.height == 0
    assert img.channels == 3
    assert img.format == ""
    assert img.annotations == []
    assert img.provenance is None


def test_provenance_record_defaults() -> None:
    prov = ProvenanceRecord(source_dataset="ds1")
    assert prov.original_class == ""
    assert prov.normalized_class == ""
    assert prov.ontology_class == ""
    assert prov.license == ""
    assert prov.dataset_version == ""
    assert prov.import_format == ""


def test_provenance_timestamp_utc() -> None:
    prov = ProvenanceRecord(source_dataset="ds1")
    parsed = datetime.fromisoformat(prov.import_timestamp)
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed) is not None


def test_raw_dataset_defaults() -> None:
    ds = RawDataset(
        dataset_id="test_yolo",
        dataset_name="test",
        import_format="yolo",
        source_path="/path",
    )
    assert ds.images == []
    assert ds.classes == []
    assert ds.metadata == {}
    assert ds.created_at is not None


def test_validation_report_defaults() -> None:
    vr = ValidationReport(dataset_id="ds1")
    assert vr.passed is False
    assert vr.missing_images == []
    assert vr.missing_annotations == []
    assert vr.total_checks == 0


def test_normalized_dataset_defaults() -> None:
    nd = NormalizedDataset(
        dataset_id="ds1",
        dataset_name="test",
    )
    assert nd.images == []
    assert nd.classes == []
    assert nd.class_mapping == {}
    assert nd.normalization_log == []


def test_ontology_mapping_report_defaults() -> None:
    omr = OntologyMappingReport(dataset_id="ds1")
    assert omr.mappings == {}
    assert omr.unmapped_classes == []
    assert omr.ontology_coverage == 0.0


def test_duplicate_entry_defaults() -> None:
    de = DuplicateEntry(
        duplicate_type="hash",
        original_id="img001",
        duplicate_id="img002",
    )
    assert de.similarity_score == 1.0
    assert de.reason == ""


def test_duplicate_report_defaults() -> None:
    dr = DuplicateReport(dataset_id="ds1")
    assert dr.duplicates == []
    assert dr.duplicate_image_count == 0
    assert dr.duplicate_ratio == 0.0


def test_harmonized_dataset_defaults() -> None:
    hd = HarmonizedDataset(dataset_id="ds1", dataset_name="test")
    assert hd.images == []
    assert hd.classes == []
    assert hd.harmonization_mapping == {}


def test_merged_dataset_defaults() -> None:
    md = MergedDataset(dataset_id="ds1", dataset_name="merged")
    assert md.source_datasets == []
    assert md.images == []
    assert md.provenance == []


def test_quality_report_defaults() -> None:
    qr = QualityReport(dataset_id="ds1")
    assert qr.quality_score == 0.0
    assert qr.duplicate_ratio == 0.0
    assert qr.passed is False
    assert qr.warnings == []


def test_quality_score_bounds() -> None:
    qr = QualityReport(
        dataset_id="ds1",
        quality_score=0.75,
        passed=True,
    )
    assert 0.0 <= qr.quality_score <= 1.0


def test_statistics_report_defaults() -> None:
    sr = StatisticsReport(dataset_id="ds1")
    assert sr.total_images == 0
    assert sr.total_annotations == 0
    assert sr.classes == []
    assert sr.objects_per_class == {}


def test_export_result_defaults() -> None:
    er = ExportResult(
        dataset_id="ds1",
        format_name="yolo",
        output_dir="/out",
    )
    assert er.image_count == 0
    assert er.annotation_count == 0


def test_import_result_defaults() -> None:
    raw = RawDataset(
        dataset_id="ds1", dataset_name="t", import_format="yolo", source_path="/p"
    )
    ir = ImportResult(
        dataset_id="ds1",
        dataset_name="t",
        import_format="yolo",
        source_path="/p",
        raw_dataset=raw,
    )
    assert ir.status == "pending"
    assert ir.validation_report is None


def test_processed_dataset_defaults() -> None:
    pd = ProcessedDataset(dataset_id="ds1", dataset_name="test")
    assert pd.version == "1.0.0"
    assert pd.status == "ready"
    assert pd.images == []
    assert pd.classes == []


def test_registry_entry_defaults() -> None:
    re = DatasetIntelligenceRegistryEntry(
        dataset_id="ds1",
        dataset_name="test",
    )
    assert re.version == "1.0.0"
    assert re.status == "pending"
    assert re.quality_score == 0.0
    assert re.processed_path == ""
