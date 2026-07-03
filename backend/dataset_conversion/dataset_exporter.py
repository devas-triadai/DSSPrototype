from __future__ import annotations

import json
import os

from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ExportResult,
)


class DatasetExporter:
    def __init__(self, config: DatasetConversionConfig | None = None) -> None:
        self._config = config or dataset_conversion_config

    async def export(
        self,
        dataset: CanonicalDataset,
        export_format: str,
        output_path: str,
    ) -> ExportResult:
        if export_format == "canonical":
            return await self._export_canonical(dataset, output_path)
        elif export_format == "coco_json":
            return await self._export_coco(dataset, output_path)
        elif export_format == "yolo_txt":
            return await self._export_yolo(dataset, output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    async def supported_export_formats(self) -> list[str]:
        return ["canonical", "coco_json", "yolo_txt"]

    async def _export_canonical(
        self,
        dataset: CanonicalDataset,
        output_path: str,
    ) -> ExportResult:
        os.makedirs(output_path, exist_ok=True)

        class_names: list[str] = []
        label_to_id: dict[str, int] = {}
        for ann in dataset.annotations:
            if ann.canonical_label not in label_to_id:
                label_to_id[ann.canonical_label] = len(label_to_id)
                class_names.append(ann.canonical_name)

        # Build COCO-like JSON for canonical export
        images_json = []
        for img in dataset.images:
            images_json.append(
                {
                    "id": img.id,
                    "file_name": os.path.basename(img.file_path) if img.file_path else img.id,
                    "width": img.width,
                    "height": img.height,
                }
            )

        annotations_json = []
        for ann in dataset.annotations:
            annotations_json.append(
                {
                    "id": ann.id,
                    "image_id": ann.image_id,
                    "category_id": label_to_id.get(ann.canonical_label, 0),
                    "bbox": [ann.x, ann.y, ann.width, ann.height],
                    "area": ann.width * ann.height,
                    "score": ann.confidence,
                    "canonical_label": ann.canonical_label,
                }
            )

        categories_json = [
            {"id": cid, "name": cname, "canonical_label": label}
            for label, cid in label_to_id.items()
            for cname in [class_names[cid]]
        ]

        output = {
            "info": {
                "dataset_name": dataset.name,
                "dataset_id": dataset.id,
                "description": f"Canonical DSS dataset: {dataset.name}",
                "ontology_version": dataset.ontology_version,
                "pipeline_version": dataset.pipeline_version,
                "source_datasets": list(dataset.source_datasets),
            },
            "images": images_json,
            "annotations": annotations_json,
            "categories": categories_json,
        }

        json_path = os.path.join(output_path, "dataset.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        total_size = os.path.getsize(json_path)
        return ExportResult(
            export_format="canonical",
            output_path=output_path,
            images_exported=dataset.image_count,
            annotations_exported=dataset.annotation_count,
            file_count=1,
            file_size_bytes=total_size,
        )

    async def _export_coco(
        self,
        dataset: CanonicalDataset,
        output_path: str,
    ) -> ExportResult:
        os.makedirs(output_path, exist_ok=True)

        label_to_id: dict[str, int] = {}
        for ann in dataset.annotations:
            if ann.canonical_label not in label_to_id:
                label_to_id[ann.canonical_label] = len(label_to_id)

        images_json = []
        for i, img in enumerate(dataset.images):
            images_json.append(
                {
                    "id": i + 1,
                    "file_name": os.path.basename(img.file_path) if img.file_path else img.id,
                    "width": img.width or 0,
                    "height": img.height or 0,
                }
            )

        img_id_map = {img.id: i + 1 for i, img in enumerate(dataset.images)}

        annotations_json = []
        for ann in dataset.annotations:
            coco_img_id = img_id_map.get(ann.image_id, 0)
            annotations_json.append(
                {
                    "id": int(hash(ann.id) % (2**31)) if ann.id else 0,
                    "image_id": coco_img_id,
                    "category_id": label_to_id.get(ann.canonical_label, 0) + 1,
                    "bbox": [ann.x, ann.y, ann.width, ann.height],
                    "area": ann.width * ann.height,
                    "score": ann.confidence,
                    "iscrowd": 0,
                }
            )

        categories_json = [
            {"id": i + 1, "name": cname, "supercategory": "object"}
            for cname, i in sorted(
                [
                    (ann.canonical_name, label_to_id[ann.canonical_label])
                    for ann in dataset.annotations
                ],
                key=lambda x: x[1],
            )
        ]
        # Deduplicate categories
        seen_cats: set[object] = set()
        deduped_categories: list[dict[str, object]] = []
        for cat in categories_json:
            if cat["id"] not in seen_cats:
                seen_cats.add(cat["id"])
                deduped_categories.append(cat)

        coco_output = {
            "info": {
                "description": f"COCO-format export of {dataset.name}",
                "version": dataset.pipeline_version,
                "year": 2026,
            },
            "images": images_json,
            "annotations": annotations_json,
            "categories": deduped_categories,
        }

        file_path = os.path.join(output_path, "coco_annotations.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, indent=2)

        total_size = os.path.getsize(file_path)
        return ExportResult(
            export_format="coco_json",
            output_path=output_path,
            images_exported=dataset.image_count,
            annotations_exported=dataset.annotation_count,
            file_count=1,
            file_size_bytes=total_size,
        )

    async def _export_yolo(
        self,
        dataset: CanonicalDataset,
        output_path: str,
    ) -> ExportResult:
        os.makedirs(output_path, exist_ok=True)
        labels_dir = os.path.join(output_path, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        label_to_id: dict[str, int] = {}
        for ann in dataset.annotations:
            if ann.canonical_label not in label_to_id:
                label_to_id[ann.canonical_label] = len(label_to_id)

        class_names_path = os.path.join(output_path, "classes.txt")
        with open(class_names_path, "w", encoding="utf-8") as f:
            for label in sorted(label_to_id, key=label_to_id.get):
                f.write(f"{label}\n")

        anns_by_img: dict[str, list[CanonicalAnnotation]] = {}
        for ann in dataset.annotations:
            anns_by_img.setdefault(ann.image_id, []).append(ann)

        total_files = 1
        for img in dataset.images:
            yolo_lines: list[str] = []
            for ann in anns_by_img.get(img.id, []):
                cx = ann.x + ann.width / 2
                cy = ann.y + ann.height / 2
                w = ann.width
                h = ann.height
                if img.width and img.height:
                    cx_norm = cx / img.width
                    cy_norm = cy / img.height
                    w_norm = w / img.width
                    h_norm = h / img.height
                else:
                    cx_norm, cy_norm, w_norm, h_norm = cx, cy, w, h
                cls_id = label_to_id.get(ann.canonical_label, 0)
                yolo_lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

            if yolo_lines:
                label_file = os.path.join(labels_dir, f"{img.id}.txt")
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                total_files += 1

        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(output_path)
            for f in fns
            if os.path.isfile(os.path.join(dp, f))
        )

        return ExportResult(
            export_format="yolo_txt",
            output_path=output_path,
            images_exported=dataset.image_count,
            annotations_exported=dataset.annotation_count,
            file_count=total_files,
            file_size_bytes=total_size,
        )
