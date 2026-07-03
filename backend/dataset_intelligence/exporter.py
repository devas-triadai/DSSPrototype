"""Dataset exporters — YOLO, COCO, Pascal VOC.

Plugin architecture: new exporters implement ``DatasetExporterInterface``
and are registered in ``ExporterRegistry``.
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.dataset_intelligence.interfaces import DatasetExporterInterface
from backend.dataset_intelligence.models import (
    ExportResult,
    HarmonizedDataset,
    MergedDataset,
    NormalizedDataset,
)

logger = logging.getLogger("dss.dataset_intelligence.exporter")


class _BaseExporter(DatasetExporterInterface):
    """Base exporter with common helpers."""

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _class_mapping_or_default(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        class_mapping: dict[str, int] | None,
    ) -> dict[str, int]:
        if class_mapping is not None:
            return class_mapping
        classes = sorted(dataset.classes)
        return {cls: i for i, cls in enumerate(classes)}


class YoloExporter(_BaseExporter):
    """Export to YOLO format (images + labels/*.txt, data.yaml)."""

    @property
    def format_name(self) -> str:
        return "yolo"

    def export(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
        splits: dict[str, list[str]] | None = None,
    ) -> ExportResult:
        logger.info("YOLO export started | dataset=%s | output=%s", dataset.dataset_id, output_dir)
        self._ensure_dir(output_dir)
        class_mapping = self._class_mapping_or_default(dataset, class_mapping)
        class_names = sorted(class_mapping.keys(), key=lambda c: class_mapping[c])

        split_sets = splits or {"train": [img.image_id for img in dataset.images]}

        for split_name, image_ids in split_sets.items():
            split_dir = output_dir / split_name
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            self._ensure_dir(img_dir)
            self._ensure_dir(lbl_dir)

            id_set = set(image_ids)
            for img in dataset.images:
                if img.image_id not in id_set:
                    continue
                src = Path(img.image_path)
                dst = img_dir / img.image_name
                if src.exists():
                    shutil.copy2(src, dst)

                lbl_path = lbl_dir / (Path(img.image_name).stem + ".txt")
                with lbl_path.open("w", encoding="utf-8") as f:
                    for ann in img.annotations:
                        cls = ann.ontology_class or ann.normalized_class or ann.class_name
                        cls_id = class_mapping.get(cls, 0)
                        x1, y1, x2, y2 = ann.bbox
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Write data.yaml
        yaml_path = output_dir / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            f.write(f"path: {output_dir}\n")
            for split_name in split_sets:
                f.write(f"{split_name}: {split_name}/images\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")

        return ExportResult(
            dataset_id=dataset.dataset_id,
            format_name=self.format_name,
            output_dir=str(output_dir),
            image_count=sum(len(ids) for ids in split_sets.values()),
            annotation_count=sum(len(img.annotations) for img in dataset.images),
            class_mapping=class_mapping,
        )


class CocoExporter(_BaseExporter):
    """Export to COCO format (images + instances.json)."""

    @property
    def format_name(self) -> str:
        return "coco"

    def export(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
        splits: dict[str, list[str]] | None = None,
    ) -> ExportResult:
        logger.info("COCO export started | dataset=%s | output=%s", dataset.dataset_id, output_dir)
        self._ensure_dir(output_dir)
        class_mapping = self._class_mapping_or_default(dataset, class_mapping)

        img_dir = output_dir / "images"
        self._ensure_dir(img_dir)

        images_meta: list[dict[str, object]] = []
        annotations = []
        ann_id = 1

        for img in dataset.images:
            src = Path(img.image_path)
            dst = img_dir / img.image_name
            if src.exists():
                shutil.copy2(src, dst)
            images_meta.append(
                {
                    "id": len(images_meta) + 1,
                    "file_name": img.image_name,
                    "width": img.width,
                    "height": img.height,
                }
            )
            for ann in img.annotations:
                cls = ann.ontology_class or ann.normalized_class or ann.class_name
                x1, y1, x2, y2 = ann.bbox
                w = x2 - x1
                h = y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": len(images_meta),
                        "category_id": class_mapping.get(cls, 0),
                        "bbox": [x1 * img.width, y1 * img.height, w * img.width, h * img.height],
                        "area": w * img.width * h * img.height,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        categories = [
            {"id": i, "name": cls} for cls, i in sorted(class_mapping.items(), key=lambda kv: kv[1])
        ]

        with (output_dir / "instances.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "images": images_meta,
                    "annotations": annotations,
                    "categories": categories,
                },
                f,
                indent=2,
            )

        return ExportResult(
            dataset_id=dataset.dataset_id,
            format_name=self.format_name,
            output_dir=str(output_dir),
            image_count=len(images_meta),
            annotation_count=len(annotations),
            class_mapping=class_mapping,
        )


class VocExporter(_BaseExporter):
    """Export to Pascal VOC format (images + XML annotations)."""

    @property
    def format_name(self) -> str:
        return "voc"

    def export(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
        splits: dict[str, list[str]] | None = None,
    ) -> ExportResult:
        logger.info("VOC export started | dataset=%s | output=%s", dataset.dataset_id, output_dir)
        self._ensure_dir(output_dir)
        class_mapping = self._class_mapping_or_default(dataset, class_mapping)

        img_dir = output_dir / "JPEGImages"
        ann_dir = output_dir / "Annotations"
        self._ensure_dir(img_dir)
        self._ensure_dir(ann_dir)

        for img in dataset.images:
            src = Path(img.image_path)
            dst = img_dir / img.image_name
            if src.exists():
                shutil.copy2(src, dst)

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = img.image_name
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img.width)
            ET.SubElement(size, "height").text = str(img.height)
            ET.SubElement(size, "depth").text = str(img.channels)

            for ann in img.annotations:
                cls = ann.ontology_class or ann.normalized_class or ann.class_name
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = cls
                bndbox = ET.SubElement(obj, "bndbox")
                x1, y1, x2, y2 = ann.bbox
                ET.SubElement(bndbox, "xmin").text = str(int(x1 * img.width))
                ET.SubElement(bndbox, "ymin").text = str(int(y1 * img.height))
                ET.SubElement(bndbox, "xmax").text = str(int(x2 * img.width))
                ET.SubElement(bndbox, "ymax").text = str(int(y2 * img.height))

            tree = ET.ElementTree(root)
            xml_path = ann_dir / (Path(img.image_name).stem + ".xml")
            tree.write(xml_path, encoding="unicode", xml_declaration=True)

        return ExportResult(
            dataset_id=dataset.dataset_id,
            format_name=self.format_name,
            output_dir=str(output_dir),
            image_count=len(dataset.images),
            annotation_count=sum(len(img.annotations) for img in dataset.images),
            class_mapping=class_mapping,
        )


class ExporterRegistry:
    """Registry for dataset exporters supporting plugin registration."""

    def __init__(self) -> None:
        self._exporters: dict[str, DatasetExporterInterface] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for exporter in [YoloExporter(), CocoExporter(), VocExporter()]:
            self._exporters[exporter.format_name] = exporter

    def register(self, exporter: DatasetExporterInterface) -> None:
        self._exporters[exporter.format_name] = exporter
        logger.info("Registered exporter: %s", exporter.format_name)

    def get(self, format_name: str) -> DatasetExporterInterface:
        exporter = self._exporters.get(format_name)
        if exporter is None:
            from backend.dataset_intelligence.exceptions import ExportError

            raise ExportError(f"No exporter registered for format: {format_name}")
        return exporter

    def list_formats(self) -> list[str]:
        return sorted(self._exporters.keys())
