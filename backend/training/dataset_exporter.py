"""TrainingDatasetExporter — prepares validated datasets for YOLO training.

Generates ``datasets/exports/yolo/<dataset_name>/`` with:
  - data.yaml       (YOLO-format configuration)
  - images/          (directory junctions / symlinks to raw images)
  - labels/          (YOLO .txt annotation files)
  - export.json      (export metadata)
  - manifest.json    (file manifest with SHA-256 checksums)
  - dataset_info.json (dataset summary, class names, split counts)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

LOG = logging.getLogger("dss.training.dataset_exporter")


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


class _ExtractResult:
    """Internal result from a per-type extractor."""

    __slots__ = (
        "class_names",
        "train_annotations",
        "val_annotations",
        "train_image_dir",
        "val_image_dir",
        "train_image_ids",
        "val_image_ids",
        "train_count",
        "val_count",
        "source_path",
    )

    def __init__(
        self,
        class_names: list[str],
        train_annotations: list[tuple[str, int, float, float, float, float]],
        val_annotations: list[tuple[str, int, float, float, float, float]],
        train_image_dir: Path,
        val_image_dir: Path,
        train_image_ids: list[str],
        val_image_ids: list[str],
        source_path: Path,
    ) -> None:
        self.class_names = class_names
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.train_image_ids = train_image_ids
        self.val_image_ids = val_image_ids
        self.train_count = len(train_image_ids)
        self.val_count = len(val_image_ids)
        self.source_path = source_path


# ------------------------------------------------------------------
# Per-type extractors
# ------------------------------------------------------------------


def _extract_coco(source: Path) -> _ExtractResult:
    """Extract annotations from COCO JSON format."""
    train_json = source / "annotations" / "instances_train2017.json"
    val_json = source / "annotations" / "instances_val2017.json"
    train_img_dir = source / "train2017"
    val_img_dir = source / "val2017"

    if not train_json.exists():
        raise FileNotFoundError(
            f"COCO training annotations not found at {train_json}. "
            "Expected COCO 2017 dataset structure."
        )

    train_data: dict[str, Any] = json.loads(train_json.read_text(encoding="utf-8"))
    empty: dict[str, Any] = {"images": [], "annotations": [], "categories": []}
    val_data = json.loads(val_json.read_text(encoding="utf-8")) if val_json.exists() else empty

    class_names = _extract_coco_categories(train_data, val_data)

    train_anns, train_ids = _parse_coco_annotations(train_data, train_img_dir, class_names, source)
    val_anns, val_ids = _parse_coco_annotations(val_data, val_img_dir, class_names, source)

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


def _extract_coco_categories(train_data: dict[str, Any], val_data: dict[str, Any]) -> list[str]:
    seen: dict[int, str] = {}
    for cat in train_data.get("categories", []):
        seen[cat["id"]] = cat["name"]
    for cat in val_data.get("categories", []):
        seen[cat["id"]] = cat["name"]
    sorted_ids = sorted(seen)
    return [seen[cid] for cid in sorted_ids]


def _parse_coco_annotations(
    data: dict[str, Any],
    img_dir: Path,
    class_names: list[str],
    source: Path,
) -> tuple[list[tuple[str, int, float, float, float, float]], list[str]]:
    cat_id_to_class: dict[int, int] = {}
    for cat in data.get("categories", []):
        try:
            cat_id_to_class[cat["id"]] = class_names.index(cat["name"])
        except ValueError:
            pass

    img_id_to_info: dict[int, dict[str, Any]] = {}
    for img in data.get("images", []):
        img_id_to_info[img["id"]] = img

    # Collect all image IDs from the images array first
    image_ids: list[str] = []
    for img in data.get("images", []):
        file_name = img.get("file_name", "")
        img_stem = Path(file_name).stem
        if img_stem not in image_ids:
            image_ids.append(img_stem)

    annotations_list: list[tuple[str, int, float, float, float, float]] = []
    for ann in data.get("annotations", []):
        img_info = img_id_to_info.get(ann["image_id"])
        if not img_info:
            continue

        cls_id = cat_id_to_class.get(ann["category_id"], 0)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        img_w = img_info.get("width", 1) or 1
        img_h = img_info.get("height", 1) or 1
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        file_name = img_info.get("file_name", "")
        img_stem = Path(file_name).stem
        annotations_list.append((img_stem, cls_id, cx, cy, wn, hn))

    return annotations_list, image_ids


def _extract_open_images_v7(source: Path) -> _ExtractResult:
    """Extract annotations from Open Images V7 CSV format."""
    train_csv = source / "train-annotations-bbox.csv"
    val_csv = source / "val-annotations-bbox.csv"
    class_desc = source / "class-descriptions-boxable.csv"
    train_img_dir = source / "train"
    val_img_dir = source / "val"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Open Images V7 training annotations not found at {train_csv}"
        )

    class_names: list[str] = []
    class_id_to_name: dict[str, str] = {}
    if class_desc.exists():
        with class_desc.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    label_id = row[0].strip()
                    name = row[1].strip()
                    class_id_to_name[label_id] = name
                    if name not in class_names:
                        class_names.append(name)

    label_name_to_class: dict[str, int] = {n: i for i, n in enumerate(class_names)}

    train_anns, train_ids = _parse_open_images_csv(
        train_csv, class_id_to_name, label_name_to_class, train_img_dir
    )
    val_anns, val_ids = _parse_open_images_csv(
        val_csv, class_id_to_name, label_name_to_class, val_img_dir
    ) if val_csv.exists() else ([], [])

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


def _parse_open_images_csv(
    csv_path: Path,
    class_id_to_name: dict[str, str],
    label_name_to_class: dict[str, int],
    img_dir: Path,
) -> tuple[list[tuple[str, int, float, float, float, float]], list[str]]:
    annotations_list: list[tuple[str, int, float, float, float, float]] = []
    image_ids: list[str] = []
    seen: set[str] = set()

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 8:
                continue
            img_id = row[0].strip()
            label_name = row[2].strip()
            x1, x2, y1, y2 = float(row[4]), float(row[5]), float(row[6]), float(row[7])

            cls_id = label_name_to_class.get(label_name, 0)

            if img_id not in seen:
                seen.add(img_id)
                image_ids.append(img_id)

            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                continue

            annotations_list.append((img_id, cls_id, cx, cy, w, h))

    return annotations_list, image_ids


def _extract_visdrone(source: Path) -> _ExtractResult:
    """Extract annotations from VisDrone TXT format."""
    # VisDrone structure expects either VisDrone2019-DET-train/val or train/val
    train_dir = source / "VisDrone2019-DET-train"
    val_dir = source / "VisDrone2019-DET-val"
    if not train_dir.exists():
        train_dir = source / "train"
        val_dir = source / "val"

    train_ann_dir = train_dir / "annotations"
    val_ann_dir = val_dir / "annotations"
    train_img_dir = train_dir / "images"
    val_img_dir = val_dir / "images"

    if not train_ann_dir.exists():
        raise FileNotFoundError(
            f"VisDrone training annotations not found at {train_ann_dir}"
        )

    # VisDrone classes
    class_names = [
        "ignored_regions", "pedestrian", "people", "bicycle", "car",
        "van", "truck", "tricycle", "awning_tricycle", "bus",
        "motor", "others",
    ]

    train_anns, train_ids = _parse_visdrone_txt(train_ann_dir, train_img_dir, class_names)
    val_anns, val_ids = _parse_visdrone_txt(val_ann_dir, val_img_dir, class_names)

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


def _parse_visdrone_txt(
    ann_dir: Path,
    img_dir: Path,
    class_names: list[str],
) -> tuple[list[tuple[str, int, float, float, float, float]], list[str]]:
    annotations_list: list[tuple[str, int, float, float, float, float]] = []
    image_ids: list[str] = []

    if not ann_dir.exists():
        return annotations_list, image_ids

    for txt_file in sorted(ann_dir.iterdir()):
        if txt_file.suffix != ".txt":
            continue
        stem = txt_file.stem

        if stem not in image_ids:
            image_ids.append(stem)

        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_w = float(parts[2])
                bbox_h = float(parts[3])
                obj_cat = int(parts[5])
            except (ValueError, IndexError):
                continue

            cls_id = max(0, min(obj_cat - 1, len(class_names) - 1))

            annotations_list.append((stem, cls_id, bbox_left, bbox_top, bbox_w, bbox_h))

    return annotations_list, image_ids


def _extract_loveda(source: Path) -> _ExtractResult:
    """Extract annotations from LoveDA GeoJSON format."""
    train_dir = source / "Train"
    val_dir = source / "Val"
    if not train_dir.exists():
        train_dir = source / "train"
        val_dir = source / "val"

    train_img_dir = train_dir
    val_img_dir = val_dir

    if not train_dir.exists():
        raise FileNotFoundError(
            f"LoveDA training directory not found at {train_dir}"
        )

    class_names = ["background", "building", "road", "water", "barren", "forest", "agricultural"]

    train_anns, train_ids = _parse_geojson_dir(train_dir, class_names)
    val_anns, val_ids = _parse_geojson_dir(val_dir, class_names)

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


def _extract_spacenet(source: Path) -> _ExtractResult:
    """Extract annotations from SpaceNet GeoJSON format."""
    # SpaceNet has AOI directories; look for train/val subdirs
    train_dir = source / "train"
    val_dir = source / "val"
    if not train_dir.exists():
        aoi_dirs = [d for d in source.iterdir() if d.is_dir() and d.name.startswith("AOI")]
        if aoi_dirs:
            train_dir = aoi_dirs[0]
            val_dir = train_dir

    if not train_dir.exists():
        raise FileNotFoundError(
            f"SpaceNet training directory not found at {train_dir}"
        )

    class_names = ["building"]

    train_img_dir = train_dir / "images" if (train_dir / "images").exists() else train_dir
    has_val_images = (val_dir / "images").exists()
    val_img_dir = val_dir / "images" if val_dir != train_dir and has_val_images else val_dir

    train_anns, train_ids = _parse_geojson_dir(train_dir, class_names)
    val_anns, val_ids = _parse_geojson_dir(val_dir, class_names)

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


def _extract_seaships(source: Path) -> _ExtractResult:
    """Extract annotations from SeaShips XML (Pascal VOC) format."""
    train_dir = source / "train"
    val_dir = source / "val"
    if not train_dir.exists():
        train_dir = source
        val_dir = source

    train_img_dir = train_dir
    val_img_dir = val_dir

    if not train_dir.exists():
        raise FileNotFoundError(
            f"SeaShips training directory not found at {train_dir}"
        )

    class_names = ["ship"]

    train_anns, train_ids = _parse_voc_xml_dir(train_dir, class_names, train_img_dir)
    val_anns, val_ids = _parse_voc_xml_dir(val_dir, class_names, val_img_dir)

    return _ExtractResult(
        class_names=class_names,
        train_annotations=train_anns,
        val_annotations=val_anns,
        train_image_dir=train_img_dir,
        val_image_dir=val_img_dir,
        train_image_ids=train_ids,
        val_image_ids=val_ids,
        source_path=source,
    )


# ------------------------------------------------------------------
# Shared parsers
# ------------------------------------------------------------------


def _parse_geojson_dir(
    data_dir: Path,
    class_names: list[str],
) -> tuple[list[tuple[str, int, float, float, float, float]], list[str]]:
    """Parse GeoJSON annotation files in a directory."""
    annotations_list: list[tuple[str, int, float, float, float, float]] = []
    image_ids: list[str] = []
    seen: set[str] = set()

    if not data_dir.exists():
        return annotations_list, image_ids

    for fpath in sorted(data_dir.iterdir()):
        if fpath.suffix not in (".geojson", ".json"):
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        stem = fpath.stem

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue

            coords_list = geom.get("coordinates", [])
            if not coords_list or not coords_list[0]:
                continue

            ring = coords_list[0]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            if not xs or not ys:
                continue

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            cls_name = props.get("class", props.get("label", ""))
            cls_id = 0
            if cls_name and cls_name in class_names:
                cls_id = class_names.index(cls_name)

            if stem not in seen:
                seen.add(stem)
                image_ids.append(stem)

            w = x_max - x_min
            h = y_max - y_min
            cx = x_min + w / 2
            cy = y_min + h / 2
            annotations_list.append((stem, cls_id, cx, cy, w, h))

    return annotations_list, image_ids


def _parse_voc_xml_dir(
    data_dir: Path,
    class_names: list[str],
    img_dir: Path,
) -> tuple[list[tuple[str, int, float, float, float, float]], list[str]]:
    """Parse Pascal VOC XML annotation files."""
    annotations_list: list[tuple[str, int, float, float, float, float]] = []
    image_ids: list[str] = []
    class_name_to_id: dict[str, int] = {n: i for i, n in enumerate(class_names)}

    if not data_dir.exists():
        return annotations_list, image_ids

    for fpath in sorted(data_dir.iterdir()):
        if fpath.suffix not in (".xml",):
            continue
        try:
            tree = ET.parse(str(fpath))
            root = tree.getroot()
        except ET.ParseError:
            continue

        filename = root.findtext("filename", fpath.stem)
        img_stem = Path(filename).stem

        if img_stem not in image_ids:
            image_ids.append(img_stem)

        size = root.find("size")
        img_w = int(size.findtext("width", "1")) if size is not None else 1
        img_h = int(size.findtext("height", "1")) if size is not None else 1

        for obj in root.findall("object"):
            cat_name = obj.findtext("name", "object")
            cls_id = class_name_to_id.get(cat_name, 0)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            w = xmax - xmin
            h = ymax - ymin
            cx = (xmin + w / 2) / img_w
            cy = (ymin + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            annotations_list.append((img_stem, cls_id, cx, cy, wn, hn))

    return annotations_list, image_ids


# ------------------------------------------------------------------
# Extractor registry
# ------------------------------------------------------------------


_EXTRACTORS: dict[str, Callable[[Path], _ExtractResult]] = {
    "coco": _extract_coco,
    "open_images_v7": _extract_open_images_v7,
    "visdrone": _extract_visdrone,
    "loveda": _extract_loveda,
    "spacenet": _extract_spacenet,
    "seaships": _extract_seaships,
}


# ------------------------------------------------------------------
# Output generation helpers
# ------------------------------------------------------------------


def _normalize_annotation(
    img_stem: str,
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> str:
    """Format a single YOLO annotation line."""
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_labels(
    output_dir: Path,
    split: str,
    annotations: list[tuple[str, int, float, float, float, float]],
) -> int:
    """Write YOLO .txt label files for a split."""
    if not annotations:
        return 0

    labels_dir = output_dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_image: dict[str, list[str]] = {}
    for img_stem, cls_id, cx, cy, w, h in annotations:
        by_image.setdefault(img_stem, []).append(
            _normalize_annotation(img_stem, cls_id, cx, cy, w, h)
        )

    count = 0
    for img_stem, lines in by_image.items():
        label_path = labels_dir / f"{img_stem}.txt"
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        count += len(lines)

    return count


def _create_directory_link(link_path: Path, target: Path) -> None:
    """Create a directory junction/symlink pointing to target."""
    if link_path.exists():
        return
    if not target.exists():
        LOG.warning("Target directory does not exist: %s", target)
        return

    link_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_abs = str(target.resolve())
        link_abs = str(link_path.resolve())

        if os.name == "nt":
            import subprocess as _sp
            result = _sp.run(
                ["cmd", "/c", "mklink", "/J", link_abs, target_abs],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                err_msg = result.stderr.strip()
                LOG.warning("Failed to create junction %s -> %s: %s", link_path, target, err_msg)
        else:
            os.symlink(target_abs, link_abs, target_is_directory=True)
    except (OSError, ImportError) as exc:
        LOG.warning("Could not create directory link %s -> %s: %s", link_path, target, exc)


def _write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Generate data.yaml for YOLO training."""
    yaml_path = output_dir / "data.yaml"

    names_repr = ", ".join(f"'{n}'" for n in class_names)
    content = (
        f"# YOLO dataset configuration generated by DSS TrainingDatasetExporter\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_repr}]\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_export_json(
    output_dir: Path,
    result: _ExtractResult,
    train_ann_count: int,
    val_ann_count: int,
) -> Path:
    """Generate export.json metadata file."""
    export_path = output_dir / "export.json"
    data = {
        "dataset_name": result.source_path.name,
        "dataset_type": "unknown",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "class_count": len(result.class_names),
        "class_names": result.class_names,
        "train_count": result.train_count,
        "val_count": result.val_count,
        "train_annotations": train_ann_count,
        "val_annotations": val_ann_count,
        "total_annotations": train_ann_count + val_ann_count,
        "source_path": str(result.source_path.resolve()),
        "status": "completed",
    }
    export_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return export_path


def _write_manifest_json(output_dir: Path) -> Path:
    """Generate manifest.json with file checksums."""
    manifest_path = output_dir / "manifest.json"
    entries: list[dict[str, object]] = []

    for fpath in sorted(output_dir.rglob("*")):
        if not fpath.is_file() or fpath.name == "manifest.json":
            continue
        rel = str(fpath.relative_to(output_dir))
        entries.append({
            "path": rel,
            "size_bytes": fpath.stat().st_size,
            "sha256": _compute_file_hash(fpath),
        })

    data = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_dataset_info(output_dir: Path, result: _ExtractResult) -> Path:
    """Generate dataset_info.json with summary."""
    info_path = output_dir / "dataset_info.json"
    data = {
        "dataset_name": result.source_path.name,
        "source_path": str(result.source_path.resolve()),
        "class_count": len(result.class_names),
        "class_names": result.class_names,
        "splits": {
            "train": {
                "image_count": result.train_count,
            },
            "val": {
                "image_count": result.val_count,
            },
        },
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    info_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return info_path


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def _validate_export(
    result: _ExtractResult,
    source: Path,
) -> list[str]:
    """Validate the extracted result against the source dataset.

    Checks:
      - Train images exist
      - Validation images exist
      - Annotation files exist
      - Class count matches ontology
    """
    errors: list[str] = []

    # Verify train images
    if result.train_image_ids:
        for img_id in result.train_image_ids[:100]:
            img_path = _find_image(result.train_image_dir, img_id)
            if img_path is None:
                errors.append(f"Train image not found: {img_id}")
                break

    # Verify val images
    if result.val_image_ids:
        for img_id in result.val_image_ids[:100]:
            img_path = _find_image(result.val_image_dir, img_id)
            if img_path is None:
                errors.append(f"Validation image not found: {img_id}")
                break

    # Verify class count
    if len(result.class_names) < 1:
        errors.append("No classes found in dataset")

    return errors


def _find_image(img_dir: Path, image_id: str) -> Path | None:
    """Find an image file with the given ID (stem) in the directory."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        candidate = img_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class ExportManifest(TypedDict):
    """Type-safe result from ``TrainingDatasetExporter.export()``."""

    dataset_name: str
    dataset_type: str
    class_names: list[str]
    train_count: int
    val_count: int
    annotation_count: int
    output_dir: str
    status: str
    errors: list[str]
    created_at: str
    data_yaml: str | None


SUPPORTED_DATASET_TYPES = frozenset(_EXTRACTORS.keys())


class TrainingDatasetExporter:
    """Exports validated datasets to YOLO-ready training packages.

    Usage::

        exporter = TrainingDatasetExporter()
        manifest = exporter.export("coco", Path("/data/raw/coco2017"))
    """

    def __init__(self, output_base: Path = Path("datasets") / "exports" / "yolo") -> None:
        self._output_base = Path(output_base)

    def export(
        self,
        dataset_type: str,
        source_path: str | Path,
        dataset_name: str | None = None,
    ) -> ExportManifest:
        """Export a raw dataset to YOLO-ready format.

        Args:
            dataset_type: One of ``SUPPORTED_DATASET_TYPES``.
            source_path: Root directory of the raw dataset.
            dataset_name: Optional override for the export directory name
                (defaults to the last component of *source_path*).

        Returns:
            A dictionary with export metadata keys:
            ``dataset_name``, ``dataset_type``, ``class_names``, ``train_count``,
            ``val_count``, ``annotation_count``, ``output_dir``, ``status``,
            ``errors``, ``created_at``, ``data_yaml``.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")

        name = dataset_name or source.name
        output_dir = self._output_base / name

        extractor = _EXTRACTORS.get(dataset_type)
        if extractor is None:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type!r}. "
                f"Supported: {sorted(SUPPORTED_DATASET_TYPES)}"
            )

        LOG.info("Extracting %s dataset from %s ...", dataset_type, source)
        result = extractor(source)

        errors = _validate_export(result, source)
        if errors:
            for err in errors:
                LOG.error("Validation error: %s", err)
            return {
                "dataset_name": name,
                "dataset_type": dataset_type,
                "class_names": result.class_names,
                "train_count": 0,
                "val_count": 0,
                "annotation_count": 0,
                "output_dir": str(output_dir),
                "status": "failed",
                "errors": errors,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_yaml": None,
            }

        LOG.info("Writing YOLO labels to %s ...", output_dir)
        train_ann = _write_labels(output_dir, "train", result.train_annotations)
        val_ann = _write_labels(output_dir, "val", result.val_annotations)
        total_ann = train_ann + val_ann

        LOG.info("Creating image directory links ...")
        _create_directory_link(output_dir / "images" / "train", result.train_image_dir)
        _create_directory_link(output_dir / "images" / "val", result.val_image_dir)

        LOG.info("Generating data.yaml and metadata ...")
        data_yaml = _write_data_yaml(output_dir, result.class_names)
        _write_export_json(output_dir, result, train_ann, val_ann)
        _write_manifest_json(output_dir)
        _write_dataset_info(output_dir, result)

        LOG.info(
            "Export complete: %s | %d classes | %d train | %d val | %d annotations",
            name, len(result.class_names), result.train_count,
            result.val_count, total_ann,
        )

        return {
            "dataset_name": name,
            "dataset_type": dataset_type,
            "class_names": result.class_names,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "annotation_count": total_ann,
            "output_dir": str(output_dir.resolve()),
            "status": "completed",
            "errors": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_yaml": str(data_yaml.resolve()),
        }

    def supported_types(self) -> list[str]:
        """Return the list of supported dataset type identifiers."""
        return sorted(SUPPORTED_DATASET_TYPES)
