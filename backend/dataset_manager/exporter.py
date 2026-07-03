"""Dataset exporter — extensible export pipeline.

Supports three formats out of the box:
  - YOLO (darknet-style .txt labels)
  - COCO (JSON with images/annotations/categories)
  - Pascal VOC (XML per image)

Open/Closed principle: add new formats by subclassing DatasetExporter
and implementing ``export()``.
"""

import json
import logging
import xml.etree.ElementTree as ET
from abc import ABC
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_manager.interfaces import DatasetExporterInterface
from backend.dataset_manager.models import DatasetExport

logger = logging.getLogger("dss.dataset_manager.exporter")


class DatasetExporter(DatasetExporterInterface, ABC):
    """Base class for all dataset exporters."""

    def __init__(self) -> None:
        self._format_name = "unknown"


class YoloExporter(DatasetExporter):
    """Exports datasets to YOLO darknet format.

    Produces:
      - images/         (symlinked or copied images)
      - labels/         (.txt files with class_id x_center y_center width height)
      - data.yaml       (class names and paths)
    """

    def __init__(self) -> None:
        super().__init__()
        self._format_name = "yolo"

    @property
    def format_name(self) -> str:
        return self._format_name

    def export(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path],
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
    ) -> DatasetExport:
        logger.info("YOLO export started: %d images, %d annotations", len(images), len(annotations))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "labels").mkdir(exist_ok=True)

        mapping = class_mapping or {}
        class_names = list(mapping.keys())

        import shutil
        for img in images:
            if img.exists():
                dest = output_dir / "images" / img.name
                shutil.copy2(str(img), str(dest))

        for ann in annotations:
            if ann.suffix.lower() == ".txt":
                dest = output_dir / "labels" / ann.name
                dest.write_text(ann.read_text())

        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images",
            "val": "images",
            "nc": len(class_names),
            "names": class_names,
        }
        (output_dir / "data.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in data_yaml.items()),
        )

        logger.info("YOLO export completed: %s", output_dir)
        return DatasetExport(
            dataset_id="",
            format_name=self._format_name,
            output_dir=str(output_dir),
            image_count=len(images),
            annotation_count=len(annotations),
            class_mapping=mapping,
        )


class CocoExporter(DatasetExporter):
    """Exports datasets to COCO JSON format.

    Produces a single ``annotations.json`` with images, annotations,
    and categories arrays.
    """

    def __init__(self) -> None:
        super().__init__()
        self._format_name = "coco"

    @property
    def format_name(self) -> str:
        return self._format_name

    def export(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path],
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
    ) -> DatasetExport:
        logger.info("COCO export started: %d images", len(images))
        output_dir.mkdir(parents=True, exist_ok=True)

        mapping = class_mapping or {}
        coco: dict[str, list[dict[str, object]]] = {
            "images": [],
            "annotations": [],
            "categories": [{"id": cid, "name": cname} for cname, cid in mapping.items()],
        }

        ann_id = 1
        for img_idx, img in enumerate(images, start=1):
            coco["images"].append({
                "id": img_idx,
                "file_name": img.name,
                "width": 0,
                "height": 0,
            })
            for ann in annotations:
                if ann.stem == img.stem:
                    try:
                        data = json.loads(ann.read_text())
                        for a in data.get("annotations", []):
                            a["id"] = ann_id
                            a["image_id"] = img_idx
                            coco["annotations"].append(a)
                            ann_id += 1
                    except Exception:
                        pass

        (output_dir / "annotations.json").write_text(
            json.dumps(coco, indent=2),
        )

        logger.info("COCO export completed: %s", output_dir)
        return DatasetExport(
            dataset_id="",
            format_name=self._format_name,
            output_dir=str(output_dir),
            image_count=len(images),
            annotation_count=ann_id - 1,
            class_mapping=mapping,
        )


class PascalVocExporter(DatasetExporter):
    """Exports datasets to Pascal VOC XML format.

    Produces one XML file per image following the VOC standard.
    """

    def __init__(self) -> None:
        super().__init__()
        self._format_name = "voc"

    @property
    def format_name(self) -> str:
        return self._format_name

    def export(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path],
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
    ) -> DatasetExport:
        logger.info("Pascal VOC export started: %d images", len(images))
        output_dir.mkdir(parents=True, exist_ok=True)

        mapping = class_mapping or {}
        id_to_name = {v: k for k, v in mapping.items()}

        for img in images:
            annotation_elem = ET.Element("annotation")
            folder = ET.SubElement(annotation_elem, "folder")
            folder.text = output_dir.name
            filename = ET.SubElement(annotation_elem, "filename")
            filename.text = img.name

            for ann in annotations:
                if ann.stem == img.stem:
                    try:
                        data = json.loads(ann.read_text())
                        for a in data.get("annotations", []):
                            obj = ET.SubElement(annotation_elem, "object")
                            name = ET.SubElement(obj, "name")
                            cat_id = a.get("category_id", 0)
                            name.text = id_to_name.get(cat_id, f"class_{cat_id}")
                            bndbox = ET.SubElement(obj, "bndbox")
                            bbox = a.get("bbox", [0, 0, 0, 0])
                            for tag, val in zip(
                                ["xmin", "ymin", "xmax", "ymax"], bbox, strict=False,
                            ):
                                el = ET.SubElement(bndbox, tag)
                                el.text = str(int(val))
                    except Exception:
                        pass

            xml_path = output_dir / f"{img.stem}.xml"
            tree = ET.ElementTree(annotation_elem)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        logger.info("Pascal VOC export completed: %s", output_dir)
        return DatasetExport(
            dataset_id="",
            format_name=self._format_name,
            output_dir=str(output_dir),
            image_count=len(images),
            annotation_count=0,
            class_mapping=mapping,
        )
