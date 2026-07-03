"""DatasetFormatResolver — translates layout metadata into canonical format strings.

The resolver bridges Dataset Catalog (which emits Storage Types like "filesystem")
and Dataset Conversion (which expects Dataset Formats like "coco_json", "yolo_txt",
"pascal_voc"). It inspects the detected layout, annotation files, and path heuristics
to determine the correct format.
"""

from __future__ import annotations

from pathlib import Path

from backend.dataset_pipeline.models import DatasetLayout


class DatasetFormatResolver:
    """Resolves dataset format from layout metadata, annotation files, and path heuristics."""

    _LAYOUT_TO_FORMAT: dict[str, str] = {
        "coco": "coco_json",
        "open_images_v7": "coco_json",
        "visdrone": "coco_json",
        "loveda": "coco_json",
        "spacenet": "coco_json",
        "seaships": "coco_json",
    }

    _EXT_TO_FORMAT: dict[str, str] = {
        ".json": "coco_json",
        ".xml": "pascal_voc",
        ".csv": "coco_json",
        ".yaml": "yolo_txt",
        ".yml": "yolo_txt",
        ".txt": "yolo_txt",
    }

    @staticmethod
    def resolve(layout: DatasetLayout | None, source_path: Path) -> str:
        """Resolve dataset format from layout and source path.

        Priority:
        1. Known layout type (COCO, OpenImages, VisDrone, etc.)
        2. Annotation file extensions in the layout
        3. Source path name heuristics (COCO, YOLO, VOC in name)
        4. Source file extension
        5. Default to ``coco_json``
        """
        if layout is not None:
            fmt = DatasetFormatResolver._LAYOUT_TO_FORMAT.get(layout.dataset_type)
            if fmt is not None:
                return fmt

            if layout.annotation_files:
                return DatasetFormatResolver._resolve_from_files(layout.annotation_files)

            if layout.annotation_directory is not None and layout.annotation_directory.is_dir():
                ann_files = list(layout.annotation_directory.iterdir())
                if ann_files:
                    return DatasetFormatResolver._resolve_from_files(ann_files)

        return DatasetFormatResolver._detect_format(source_path)

    @staticmethod
    def _resolve_from_files(files: list[Path]) -> str:
        for f in files:
            ext = f.suffix.lower()
            fmt = DatasetFormatResolver._EXT_TO_FORMAT.get(ext)
            if fmt is not None:
                return fmt
        return "coco_json"

    @staticmethod
    def _detect_format(source_path: Path) -> str:
        name_lower = source_path.name.lower()
        if "coco" in name_lower:
            return "coco_json"
        if "yolo" in name_lower or "darknet" in name_lower:
            return "yolo_txt"
        if "voc" in name_lower or "pascal" in name_lower:
            return "pascal_voc"
        if source_path.is_file():
            ext = source_path.suffix.lower()
            fmt = DatasetFormatResolver._EXT_TO_FORMAT.get(ext)
            if fmt is not None:
                return fmt
        return "coco_json"
