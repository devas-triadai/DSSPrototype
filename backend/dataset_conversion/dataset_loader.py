from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.exceptions import LoadError
from backend.dataset_conversion.interfaces import DatasetLoaderInterface
from backend.dataset_conversion.models import (
    CoordinateSystem,
    DatasetFormat,
    GeometryType,
    ImageInfo,
    LoadResult,
    SourceAnnotation,
    SourceCategory,
)


class DatasetLoader(DatasetLoaderInterface):
    def __init__(self, config: DatasetConversionConfig | None = None) -> None:
        self._config = config or dataset_conversion_config

    async def load(
        self,
        path: str,
        dataset_format: str,
        **kwargs: str,
    ) -> LoadResult:
        fmt_enum = self._resolve_format(dataset_format)
        if fmt_enum == DatasetFormat.COCO_JSON:
            return await self._load_coco_json(path, **kwargs)
        elif fmt_enum == DatasetFormat.YOLO_TXT:
            return await self._load_yolo_txt(path, **kwargs)
        elif fmt_enum == DatasetFormat.PASCAL_VOC:
            return await self._load_pascal_voc(path, **kwargs)
        elif fmt_enum == DatasetFormat.OPEN_IMAGES_CSV:
            return await self._load_open_images_csv(path, **kwargs)
        elif fmt_enum == DatasetFormat.CANONICAL:
            return await self._load_canonical(path)
        else:
            raise LoadError(f"Unsupported dataset format: {dataset_format}")

    async def load_image(self, image_id: str) -> ImageInfo | None:
        return None

    async def supported_formats(self) -> list[str]:
        return [f.value for f in DatasetFormat]

    def _resolve_format(self, fmt: str) -> DatasetFormat:
        try:
            return DatasetFormat(fmt)
        except ValueError:
            raise LoadError(f"Unknown dataset format: {fmt}")

    async def _load_coco_json(self, path: str, **kwargs: str) -> LoadResult:
        data_dir = kwargs.get("data_dir", os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                coco = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise LoadError(f"Failed to load COCO JSON from {path}: {e}")

        images: list[ImageInfo] = []
        for img in coco.get("images", []):
            file_name = img.get("file_name", "")
            images.append(
                ImageInfo(
                    id=str(img["id"]),
                    file_path=str(Path(data_dir) / file_name) if data_dir else file_name,
                    width=img.get("width"),
                    height=img.get("height"),
                    format=os.path.splitext(file_name)[1].lstrip(".") if "." in file_name else None,
                )
            )

        categories: list[SourceCategory] = []
        for cat in coco.get("categories", []):
            categories.append(
                SourceCategory(
                    id=cat["id"],
                    name=cat["name"],
                    supercategory=cat.get("supercategory"),
                )
            )

        cat_id_to_name: dict[int | str, str] = {}
        for cat in categories:
            cat_id_to_name[cat.id] = cat.name

        annotations: list[SourceAnnotation] = []
        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            annotations.append(
                SourceAnnotation(
                    id=str(ann["id"]),
                    image_id=str(ann["image_id"]),
                    category_id=ann["category_id"],
                    category_name=cat_id_to_name.get(ann["category_id"], "unknown"),
                    geometry_type=GeometryType.BBOX,
                    coordinates=tuple(bbox),
                    coordinate_system=CoordinateSystem.PIXEL,
                    confidence=ann.get("score"),
                    image_width=coco_img.get("width") if (coco_img := next(
                        (x for x in coco.get("images", []) if x["id"] == ann["image_id"]), None
                    )) else None,
                    image_height=coco_img.get("height") if coco_img else None,
                    metadata={
                        "area": str(ann.get("area", "")),
                        "iscrowd": str(ann.get("iscrowd", 0)),
                    },
                )
            )

        return LoadResult(
            dataset_name=kwargs.get("dataset_name", Path(path).stem),
            source_path=path,
            dataset_format=DatasetFormat.COCO_JSON.value,
            images=tuple(images),
            annotations=tuple(annotations),
            categories=tuple(categories),
            image_count=len(images),
            annotation_count=len(annotations),
            category_count=len(categories),
        )

    async def _load_yolo_txt(self, path: str, **kwargs: str) -> LoadResult:
        data_dir = kwargs.get("data_dir", os.path.dirname(path))
        names_path = kwargs.get("names_path", "")
        dataset_name = kwargs.get("dataset_name", Path(path).stem)

        categories: list[SourceCategory] = []
        if names_path and os.path.isfile(names_path):
            with open(names_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        categories.append(SourceCategory(id=i, name=name))
        else:
            categories.append(SourceCategory(id=0, name="object"))

        cat_id_to_name = {c.id: c.name for c in categories}

        annotations: list[SourceAnnotation] = []
        image_paths: list[str] = []
        if os.path.isdir(path):
            label_dir = path
            image_dir = kwargs.get("image_dir", data_dir)
            for fname in os.listdir(label_dir):
                if fname.endswith(".txt"):
                    stem = os.path.splitext(fname)[0]
                    image_file = self._find_image(stem, image_dir)
                    if image_file:
                        image_paths.append(image_file)
                    full_path = os.path.join(label_dir, fname)
                    with open(full_path, encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cat_id = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])
                                annotations.append(
                                    SourceAnnotation(
                                        id=f"{stem}_{len(annotations)}",
                                        image_id=stem,
                                        category_id=cat_id,
                                        category_name=cat_id_to_name.get(cat_id, "unknown"),
                                        geometry_type=GeometryType.NORMALIZED,
                                        coordinates=(cx, cy, w, h),
                                        coordinate_system=CoordinateSystem.NORMALIZED,
                                    )
                                )

        images: list[ImageInfo] = []
        for img_path in image_paths:
            images.append(
                ImageInfo(
                    id=Path(img_path).stem,
                    file_path=img_path,
                )
            )

        return LoadResult(
            dataset_name=dataset_name,
            source_path=path,
            dataset_format=DatasetFormat.YOLO_TXT.value,
            images=tuple(images),
            annotations=tuple(annotations),
            categories=tuple(categories),
            image_count=len(images),
            annotation_count=len(annotations),
            category_count=len(categories),
        )

    async def _load_pascal_voc(self, path: str, **kwargs: str) -> LoadResult:
        data_dir = kwargs.get("data_dir", os.path.dirname(path))
        dataset_name = kwargs.get("dataset_name", Path(path).stem)

        annotations: list[SourceAnnotation] = []
        categories_seen: dict[str, int] = {}

        if os.path.isdir(path):
            xml_files = [f for f in os.listdir(path) if f.endswith(".xml")]
        else:
            xml_files = [path]

        for xml_file in xml_files:
            xml_path = os.path.join(path, xml_file) if os.path.isdir(path) else xml_file
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError as e:
                raise LoadError(f"Failed to parse VOC XML {xml_file}: {e}")

            filename = root.findtext("filename", "")
            img_id = os.path.splitext(filename)[0] if filename else Path(xml_file).stem
            size = root.find("size")
            img_w = int(size.findtext("width", "0")) if size is not None else 0
            img_h = int(size.findtext("height", "0")) if size is not None else 0

            for obj in root.findall("object"):
                cat_name = obj.findtext("name", "unknown")
                if cat_name not in categories_seen:
                    categories_seen[cat_name] = len(categories_seen)
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.findtext("xmin", "0"))
                    ymin = float(bndbox.findtext("ymin", "0"))
                    xmax = float(bndbox.findtext("xmax", "0"))
                    ymax = float(bndbox.findtext("ymax", "0"))
                    annotations.append(
                        SourceAnnotation(
                            id=f"{img_id}_{len(annotations)}",
                            image_id=img_id,
                            category_id=categories_seen[cat_name],
                            category_name=cat_name,
                            geometry_type=GeometryType.BBOX,
                            coordinates=(xmin, ymin, xmax - xmin, ymax - ymin),
                            coordinate_system=CoordinateSystem.PIXEL,
                            image_width=img_w if img_w > 0 else None,
                            image_height=img_h if img_h > 0 else None,
                        )
                    )

        categories = [SourceCategory(id=v, name=k) for k, v in categories_seen.items()]

        images: list[ImageInfo] = []
        seen_ids: set[str] = set()
        for ann in annotations:
            if ann.image_id not in seen_ids and isinstance(ann.image_id, str):
                seen_ids.add(ann.image_id)
                images.append(
                    ImageInfo(
                        id=ann.image_id,
                        file_path=str(Path(data_dir) / f"{ann.image_id}.jpg"),
                        width=ann.image_width,
                        height=ann.image_height,
                    )
                )

        return LoadResult(
            dataset_name=dataset_name,
            source_path=path,
            dataset_format=DatasetFormat.PASCAL_VOC.value,
            images=tuple(images),
            annotations=tuple(annotations),
            categories=tuple(categories),
            image_count=len(images),
            annotation_count=len(annotations),
            category_count=len(categories),
        )

    async def _load_open_images_csv(self, path: str, **kwargs: str) -> LoadResult:
        data_dir = kwargs.get("data_dir", os.path.dirname(path))
        dataset_name = kwargs.get("dataset_name", Path(path).stem)
        class_names_path = kwargs.get("class_names_path", "")

        categories: list[SourceCategory] = []
        if class_names_path and os.path.isfile(class_names_path):
            with open(class_names_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        categories.append(
                            SourceCategory(id=parts[0].strip(), name=parts[1].strip())
                        )

        cat_id_to_name = {c.id: c.name for c in categories}

        annotations: list[SourceAnnotation] = []
        image_ids_in_anns: set[str] = set()

        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 8:
                        img_id = parts[0]
                        label_name = parts[2]
                        conf_str = parts[3]
                        x1, x2, y1, y2 = map(float, parts[4:8])
                        cat_id_str = label_name
                        image_ids_in_anns.add(img_id)
                        annotations.append(
                            SourceAnnotation(
                                id=f"{img_id}_{len(annotations)}",
                                image_id=img_id,
                                category_id=cat_id_str,
                                category_name=label_name,
                                geometry_type=GeometryType.BBOX,
                                coordinates=(x1, y1, x2 - x1, y2 - y1),
                                coordinate_system=CoordinateSystem.PIXEL,
                                confidence=float(conf_str)
                                if conf_str and conf_str != "1"
                                else None,
                                metadata={"confidence_raw": conf_str} if conf_str else {},
                            )
                        )

        if not categories:
            seen_cats: set[str] = set()
            for ann in annotations:
                if str(ann.category_id) not in seen_cats:
                    seen_cats.add(str(ann.category_id))
                    categories.append(
                        SourceCategory(
                            id=str(ann.category_id),
                            name=cat_id_to_name.get(str(ann.category_id), str(ann.category_id)),
                        )
                    )

        images: list[ImageInfo] = []
        for img_id in sorted(image_ids_in_anns):
            images.append(
                ImageInfo(
                    id=img_id,
                    file_path=str(Path(data_dir) / f"{img_id}.jpg"),
                )
            )

        return LoadResult(
            dataset_name=dataset_name,
            source_path=path,
            dataset_format=DatasetFormat.OPEN_IMAGES_CSV.value,
            images=tuple(images),
            annotations=tuple(annotations),
            categories=tuple(categories),
            image_count=len(images),
            annotation_count=len(annotations),
            category_count=len(categories),
        )

    async def _load_canonical(self, path: str) -> LoadResult:
        json_path = os.path.join(path, "dataset.json")
        if os.path.isfile(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            images: list[ImageInfo] = []
            for img in data.get("images", []):
                images.append(ImageInfo(**img))
            annotations: list[SourceAnnotation] = []
            categories: list[SourceCategory] = []
            for cat in data.get("categories", []):
                categories.append(SourceCategory(**cat))
            return LoadResult(
                dataset_name=data.get("name", Path(path).stem),
                source_path=path,
                dataset_format=DatasetFormat.CANONICAL.value,
                images=tuple(images),
                annotations=tuple(annotations),
                categories=tuple(categories),
                image_count=len(images),
                annotation_count=0,
                category_count=len(categories),
            )
        raise LoadError(f"Canonical dataset not found at {path}")

    def _find_image(self, stem: str, image_dir: str) -> str | None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = os.path.join(image_dir, f"{stem}{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None
