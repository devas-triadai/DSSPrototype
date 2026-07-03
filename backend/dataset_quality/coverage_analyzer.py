from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.interfaces import CoverageAnalyzerInterface
from backend.dataset_quality.models import (
    CoverageAnalysisResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class CoverageAnalyzer(CoverageAnalyzerInterface):
    async def analyze(
        self,
        dataset: CanonicalDataset,
        ontology_classes: Sequence[str] | None = None,
    ) -> CoverageAnalysisResult:
        issues: list[QualityIssue] = []

        found_classes: set[str] = set()
        for ann in dataset.annotations:
            found_classes.add(ann.canonical_label)

        ontology_set: set[str] = set(ontology_classes) if ontology_classes else found_classes
        total_ontology = len(ontology_set)

        if total_ontology > 0:
            covered = len(found_classes & ontology_set)
            ontology_coverage_pct = (covered / total_ontology) * 100.0
        else:
            covered = 0
            ontology_coverage_pct = 100.0 if len(found_classes) == 0 else 0.0

        missing = sorted(ontology_set - found_classes) if ontology_classes else []

        for cls in missing:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.COVERAGE,
                    message=f"Ontology class '{cls}' is not covered in the dataset",
                    location=cls,
                    suggestion="Collect samples for this class or adjust ontology expectations",
                    details={"class": cls},
                )
            )

        total_images = dataset.image_count
        annotated_image_ids = {ann.image_id for ann in dataset.annotations}
        annotated_count = len(annotated_image_ids)
        image_coverage_pct = (annotated_count / total_images * 100) if total_images > 0 else 0.0

        scene_diversity = 0.0
        if total_images > 0:
            unique_paths = {img.file_path.split("/")[-1].split("_")[0] for img in dataset.images}
            scene_diversity = min(1.0, len(unique_paths) / total_images) if unique_paths else 0.0

        object_diversity = 0.0
        if dataset.annotation_count > 0 and len(found_classes) > 0:
            anns_per_class = dataset.annotation_count / len(found_classes)
            object_diversity = min(1.0, anns_per_class / 100)

        completeness_pct = (
            ontology_coverage_pct * 0.4
            + image_coverage_pct * 0.3
            + scene_diversity * 100 * 0.15
            + object_diversity * 100 * 0.15
        )

        if ontology_coverage_pct < 50 and ontology_classes:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.COVERAGE,
                    message=f"Ontology coverage is only {ontology_coverage_pct:.1f}%",
                    location="dataset",
                    suggestion="Expand dataset to cover more ontology classes",
                    details={
                        "coverage_pct": ontology_coverage_pct,
                        "covered": covered,
                        "total": total_ontology,
                    },
                )
            )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return CoverageAnalysisResult(
            ontology_coverage_pct=ontology_coverage_pct,
            image_coverage_pct=image_coverage_pct,
            scene_diversity=scene_diversity,
            object_diversity=object_diversity,
            completeness_pct=completeness_pct,
            covered_classes=covered,
            total_ontology_classes=total_ontology,
            missing_classes=tuple(missing),
            issues=tuple(issues),
            passed=passed,
        )
