"""Pluggable format parser registry and concrete parsers.

Supports YOLO, COCO, Pascal VOC, CSV, JSON, and GeoJSON out of the box.
New formats are added by implementing ``FormatParserInterface`` and
registering the parser with ``FormatParserRegistry``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from backend.dataset_intelligence.exceptions import FormatDetectionError
from backend.dataset_intelligence.interfaces import (
    FormatParserInterface,
    FormatParserRegistryInterface,
)
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    ProvenanceRecord,
    RawDataset,
)

logger = logging.getLogger("dss.dataset_intelligence.parser")


def _compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _bbox_to_xyxy_normalized(
    bbox: tuple[float, float, float, float],
    fmt: str,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert any bbox format to xyxy normalized."""
    if fmt == "xyxy_normalized":
        return bbox
    if fmt == "xyxy_absolute":
        x1, y1, x2, y2 = bbox
        return (x1 / width, y1 / height, x2 / width, y2 / height)
    if fmt == "xywh_normalized":
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    if fmt == "xywh_absolute":
        x, y, w, h = bbox
        return (x / width, y / height, (x + w) / width, (y + h) / height)
    if fmt == "cxcywh_normalized":
        cx, cy, w, h = bbox
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    if fmt == "cxcywh_absolute":
        cx, cy, w, h = bbox
        return (
            (cx - w / 2) / width,
            (cy - h / 2) / height,
            (cx + w / 2) / width,
            (cy + h / 2) / height,
        )
    raise ValueError(f"Unsupported bbox format: {fmt}")


class _BaseParser(FormatParserInterface):
    """Base parser with common helper methods."""

    def _build_image_record(
        self,
        image_path: Path,
        annotations: list[Annotation],
        source_dataset: str,
        import_format: str,
    ) -> ImageRecord:
        width, height = 0, 0
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as img:
                width, height = img.size
        except Exception:
            logger.warning("Cannot read image dimensions for %s", image_path)

        return ImageRecord(
            image_id=image_path.stem,
            image_path=str(image_path),
            image_name=image_path.name,
            width=width,
            height=height,
            format=image_path.suffix.lstrip(".").lower(),
            annotations=annotations,
            checksum=_compute_checksum(image_path),
            provenance=ProvenanceRecord(
                source_dataset=source_dataset,
                original_path=str(image_path),
                import_format=import_format,
            ),
        )


class YoloParser(_BaseParser):
    """Parse YOLO-format datasets (images + .txt annotations)."""

    @property
    def supported_formats(self) -> list[str]:
        return ["yolo"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("YOLO parser started: %s", source_path)
        images_dir = source_path / "images"
        labels_dir = source_path / "labels"

        if not images_dir.exists():
            images_dir = source_path
        if not labels_dir.exists():
            labels_dir = source_path

        image_paths = sorted(
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        classes_file = source_path / "classes.txt"
        class_names: list[str] = []
        if classes_file.exists():
            class_names = [
                line.strip() for line in classes_file.read_text().splitlines() if line.strip()
            ]

        images: list[ImageRecord] = []
        for img_path in image_paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            annotations: list[Annotation] = []
            if label_path.exists():
                try:
                    with label_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            class_name = (
                                class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                            )
                            bbox = _bbox_to_xyxy_normalized(
                                (cx, cy, w, h),
                                "cxcywh_normalized",
                                img_path.stat().st_size,
                                img_path.stat().st_size,
                            )
                            # We need width/height; use placeholder since we already computed above
                            # Re-parse with actual dimensions
                            width, height = 0, 0
                            try:
                                from PIL import Image as PILImage

                                with PILImage.open(img_path) as img:
                                    width, height = img.size
                            except Exception:
                                pass
                            bbox = _bbox_to_xyxy_normalized(
                                (cx, cy, w, h), "cxcywh_normalized", width, height
                            )
                            annotations.append(
                                Annotation(
                                    class_name=class_name,
                                    bbox=bbox,
                                    bbox_format="xyxy_normalized",
                                )
                            )
                except Exception as exc:
                    logger.warning("Failed to parse YOLO label %s: %s", label_path, exc)

            images.append(self._build_image_record(img_path, annotations, source_path.name, "yolo"))

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"yolo_{source_path.name}",
            dataset_name=source_path.name,
            import_format="yolo",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class CocoParser(_BaseParser):
    """Parse COCO-format datasets (instances.json)."""

    @property
    def supported_formats(self) -> list[str]:
        return ["coco"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("COCO parser started: %s", source_path)
        ann_file = next(
            (
                f
                for f in source_path.iterdir()
                if f.name.endswith("annotations.json") or f.name == "instances.json"
            ),
            None,
        )
        if ann_file is None:
            ann_file = next((f for f in source_path.rglob("*.json")), None)
        if ann_file is None:
            raise FormatDetectionError(f"No COCO annotation file found in {source_path}")

        with ann_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images_dir = source_path / "images"
        if not images_dir.exists():
            images_dir = source_path

        cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
        img_id_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        images: list[ImageRecord] = []
        for img_info in coco.get("images", []):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_path = images_dir / file_name
            if not img_path.exists():
                img_path = source_path / file_name

            width = img_info.get("width", 0)
            height = img_info.get("height", 0)

            annotations: list[Annotation] = []
            for ann in img_id_to_anns.get(img_id, []):
                bbox = ann.get("bbox", [0, 0, 0, 0])
                x, y, w, h = bbox
                cat_id = ann.get("category_id", -1)
                class_name = cat_id_to_name.get(cat_id, str(cat_id))
                bbox_norm = _bbox_to_xyxy_normalized((x, y, w, h), "xywh_absolute", width, height)
                annotations.append(
                    Annotation(
                        class_name=class_name,
                        bbox=bbox_norm,
                        bbox_format="xyxy_normalized",
                        segmentation=ann.get("segmentation"),
                    )
                )

            if img_path.exists():
                images.append(
                    self._build_image_record(img_path, annotations, source_path.name, "coco")
                )
            else:
                logger.warning("COCO image not found: %s", img_path)

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"coco_{source_path.name}",
            dataset_name=source_path.name,
            import_format="coco",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class VocParser(_BaseParser):
    """Parse Pascal VOC-format datasets (XML annotations)."""

    @property
    def supported_formats(self) -> list[str]:
        return ["voc"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("VOC parser started: %s", source_path)
        xml_files = list(source_path.rglob("*.xml"))
        images_dir = source_path / "JPEGImages"
        if not images_dir.exists():
            images_dir = source_path / "images"
        if not images_dir.exists():
            images_dir = source_path

        images: list[ImageRecord] = []
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.findtext("filename", xml_file.stem + ".jpg")
                img_path = images_dir / filename
                if not img_path.exists():
                    img_path = source_path / filename

                size = root.find("size")
                width = int(size.findtext("width", "0")) if size is not None else 0
                height = int(size.findtext("height", "0")) if size is not None else 0

                annotations: list[Annotation] = []
                for obj in root.findall("object"):
                    name = obj.findtext("name", "unknown")
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = float(bndbox.findtext("xmin", "0"))
                        ymin = float(bndbox.findtext("ymin", "0"))
                        xmax = float(bndbox.findtext("xmax", "0"))
                        ymax = float(bndbox.findtext("ymax", "0"))
                        bbox_norm = _bbox_to_xyxy_normalized(
                            (xmin, ymin, xmax, ymax), "xyxy_absolute", width, height
                        )
                        annotations.append(
                            Annotation(
                                class_name=name,
                                bbox=bbox_norm,
                                bbox_format="xyxy_normalized",
                            )
                        )

                if img_path.exists():
                    images.append(
                        self._build_image_record(img_path, annotations, source_path.name, "voc")
                    )
                else:
                    logger.warning("VOC image not found: %s", img_path)
            except Exception as exc:
                logger.warning("Failed to parse VOC XML %s: %s", xml_file, exc)

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"voc_{source_path.name}",
            dataset_name=source_path.name,
            import_format="voc",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class CsvParser(_BaseParser):
    """Parse CSV-format datasets (one row per annotation)."""

    @property
    def supported_formats(self) -> list[str]:
        return ["csv"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("CSV parser started: %s", source_path)
        csv_files = list(source_path.glob("*.csv"))
        if not csv_files:
            raise FormatDetectionError(f"No CSV file found in {source_path}")
        csv_file = csv_files[0]

        images_dir = source_path / "images"
        if not images_dir.exists():
            images_dir = source_path

        img_id_to_anns: dict[str, list[Annotation]] = {}
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("image_filename") or row.get("filename") or row.get("image")
                if not filename:
                    continue
                class_name = (
                    row.get("class_name") or row.get("class") or row.get("label", "unknown")
                )
                x1 = float(row.get("x_min", row.get("xmin", 0)))
                y1 = float(row.get("y_min", row.get("ymin", 0)))
                x2 = float(row.get("x_max", row.get("xmax", 0)))
                y2 = float(row.get("y_max", row.get("ymax", 0)))
                width = int(row.get("width", 0))
                height = int(row.get("height", 0))
                bbox_norm = _bbox_to_xyxy_normalized(
                    (x1, y1, x2, y2), "xyxy_absolute", width, height
                )
                img_id_to_anns.setdefault(filename, []).append(
                    Annotation(
                        class_name=class_name,
                        bbox=bbox_norm,
                        bbox_format="xyxy_normalized",
                    )
                )

        images: list[ImageRecord] = []
        for filename, anns in img_id_to_anns.items():
            img_path = images_dir / filename
            if img_path.exists():
                images.append(
                    self._build_image_record(img_path, anns, source_path.name, "csv")
                )
            else:
                logger.warning("CSV image not found: %s", img_path)

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"csv_{source_path.name}",
            dataset_name=source_path.name,
            import_format="csv",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class JsonParser(_BaseParser):
    """Parse generic JSON-format datasets."""

    @property
    def supported_formats(self) -> list[str]:
        return ["json"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("JSON parser started: %s", source_path)
        json_files = [f for f in source_path.iterdir() if f.suffix == ".json"]
        if not json_files:
            raise FormatDetectionError(f"No JSON file found in {source_path}")
        json_file = json_files[0]

        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images_dir = source_path / "images"
        if not images_dir.exists():
            images_dir = source_path

        images: list[ImageRecord] = []
        raw_entries: list[dict[str, Any]] = (
            data if isinstance(data, list) else data.get("images", [])
        )
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename") or entry.get("file_name", "")
            img_path = images_dir / filename
            width = entry.get("width", 0)
            height = entry.get("height", 0)
            annotations: list[Annotation] = []
            for ann in entry.get("annotations", []):
                class_name = ann.get("class", ann.get("class_name", "unknown"))
                bbox = ann.get("bbox", [0, 0, 0, 0])
                fmt = ann.get("bbox_format", "xyxy_normalized")
                bbox_norm = _bbox_to_xyxy_normalized(tuple(bbox), fmt, width, height)
                annotations.append(
                    Annotation(
                        class_name=class_name,
                        bbox=bbox_norm,
                        bbox_format="xyxy_normalized",
                    )
                )
            if img_path.exists():
                images.append(
                    self._build_image_record(img_path, annotations, source_path.name, "json")
                )
            else:
                logger.warning("JSON image not found: %s", img_path)

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"json_{source_path.name}",
            dataset_name=source_path.name,
            import_format="json",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class GeoJsonParser(_BaseParser):
    """Parse GeoJSON-format datasets."""

    @property
    def supported_formats(self) -> list[str]:
        return ["geojson"]

    def parse(self, source_path: Path) -> RawDataset:
        logger.info("GeoJSON parser started: %s", source_path)
        geojson_files = [f for f in source_path.iterdir() if f.suffix == ".geojson"]
        if not geojson_files:
            raise FormatDetectionError(f"No GeoJSON file found in {source_path}")
        geojson_file = geojson_files[0]

        with geojson_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images_dir = source_path / "images"
        if not images_dir.exists():
            images_dir = source_path

        img_id_to_anns: dict[str, list[Annotation]] = {}
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            filename = props.get("image_filename") or props.get("filename", "")
            class_name = props.get("class") or props.get("class_name", "unknown")
            bbox = props.get("bbox", [0, 0, 0, 0])
            width = props.get("width", 0)
            height = props.get("height", 0)
            fmt = props.get("bbox_format", "xyxy_normalized")
            bbox_norm = _bbox_to_xyxy_normalized(tuple(bbox), fmt, width, height)
            img_id_to_anns.setdefault(filename, []).append(
                Annotation(
                    class_name=class_name,
                    bbox=bbox_norm,
                    bbox_format="xyxy_normalized",
                )
            )

        images: list[ImageRecord] = []
        for filename, anns in img_id_to_anns.items():
            img_path = images_dir / filename
            if img_path.exists():
                images.append(
                    self._build_image_record(img_path, anns, source_path.name, "geojson")
                )
            else:
                logger.warning("GeoJSON image not found: %s", img_path)

        all_classes = sorted({ann.class_name for img in images for ann in img.annotations})
        return RawDataset(
            dataset_id=f"geojson_{source_path.name}",
            dataset_name=source_path.name,
            import_format="geojson",
            source_path=str(source_path),
            images=images,
            classes=all_classes,
        )


class FormatParserRegistry(FormatParserRegistryInterface):
    """Registry for format parsers with auto-detection support."""

    def __init__(self) -> None:
        self._parsers: dict[str, FormatParserInterface] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for parser in [
            YoloParser(),
            CocoParser(),
            VocParser(),
            CsvParser(),
            JsonParser(),
            GeoJsonParser(),
        ]:
            self.register(parser)

    def register(self, parser: FormatParserInterface) -> None:
        for fmt in parser.supported_formats:
            self._parsers[fmt] = parser
            logger.debug("Registered parser for format: %s", fmt)

    def get_parser(self, format_name: str) -> FormatParserInterface:
        parser = self._parsers.get(format_name.lower())
        if parser is None:
            raise FormatDetectionError(f"No parser registered for format: {format_name}")
        return parser

    def detect_format(self, source_path: Path) -> str:
        if not source_path.exists():
            raise FormatDetectionError(f"Source path does not exist: {source_path}")

        # YOLO: data.yaml or images/labels structure
        if (source_path / "data.yaml").exists():
            return "yolo"
        if (source_path / "images").exists() and (source_path / "labels").exists():
            return "yolo"

        # COCO: annotations.json or instances.json
        if any(
            f.name.endswith("annotations.json") or f.name == "instances.json"
            for f in source_path.iterdir()
        ):
            return "coco"

        # VOC: XML files
        if any(f.suffix == ".xml" for f in source_path.rglob("*")):
            return "voc"

        # GeoJSON: .geojson files
        if any(f.suffix == ".geojson" for f in source_path.iterdir()):
            return "geojson"

        # CSV: .csv files
        if any(f.suffix == ".csv" for f in source_path.iterdir()):
            return "csv"

        # JSON: .json files (lowest priority)
        if any(f.suffix == ".json" for f in source_path.iterdir()):
            return "json"

        raise FormatDetectionError(f"Could not detect format for {source_path}")
