from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from backend.dataset_conversion.exceptions import AnnotationError
from backend.dataset_conversion.interfaces import AnnotationLoaderInterface
from backend.dataset_conversion.models import (
    CoordinateSystem,
    GeometryType,
    SourceAnnotation,
    SourceCategory,
)


class AnnotationLoader(AnnotationLoaderInterface):
    async def parse_annotations(
        self,
        data: str,
        dataset_format: str,
    ) -> list[SourceAnnotation]:
        if dataset_format == "coco_json":
            return await self._parse_coco_annotations(data)
        elif dataset_format == "yolo_txt":
            return await self._parse_yolo_annotations(data)
        elif dataset_format == "pascal_voc":
            return await self._parse_voc_annotations(data)
        elif dataset_format == "open_images_csv":
            return await self._parse_open_images_annotations(data)
        else:
            raise AnnotationError(f"Unsupported annotation format: {dataset_format}")

    async def parse_categories(
        self,
        data: str,
        dataset_format: str,
    ) -> list[SourceCategory]:
        if dataset_format == "coco_json":
            return await self._parse_coco_categories(data)
        elif dataset_format == "pascal_voc":
            return await self._parse_voc_categories(data)
        elif dataset_format == "yolo_txt":
            return self._parse_yolo_categories(data)
        else:
            return []

    async def _parse_coco_annotations(self, data: str) -> list[SourceAnnotation]:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise AnnotationError(f"Invalid COCO JSON: {e}")

        cat_map: dict[int, str] = {}
        for cat in parsed.get("categories", []):
            cat_map[cat["id"]] = cat["name"]

        results: list[SourceAnnotation] = []
        for ann in parsed.get("annotations", []):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            seg = ann.get("segmentation")
            geom_type = GeometryType.SEGMENTATION if seg and len(seg) > 0 else GeometryType.BBOX
            results.append(
                SourceAnnotation(
                    id=str(ann["id"]),
                    image_id=str(ann["image_id"]),
                    category_id=ann["category_id"],
                    category_name=cat_map.get(ann["category_id"], "unknown"),
                    geometry_type=geom_type,
                    coordinates=tuple(bbox),
                    coordinate_system=CoordinateSystem.PIXEL,
                    confidence=ann.get("score"),
                    metadata={"iscrowd": str(ann.get("iscrowd", 0))},
                )
            )
        return results

    async def _parse_coco_categories(self, data: str) -> list[SourceCategory]:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise AnnotationError(f"Invalid COCO JSON: {e}")
        return [
            SourceCategory(id=cat["id"], name=cat["name"], supercategory=cat.get("supercategory"))
            for cat in parsed.get("categories", [])
        ]

    async def _parse_yolo_annotations(self, data: str) -> list[SourceAnnotation]:
        lines = [ln.strip() for ln in data.split("\n") if ln.strip()]
        results: list[SourceAnnotation] = []
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 5:
                cat_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                results.append(
                    SourceAnnotation(
                        id=str(i),
                        image_id="",
                        category_id=cat_id,
                        category_name=str(cat_id),
                        geometry_type=GeometryType.NORMALIZED,
                        coordinates=(cx, cy, w, h),
                        coordinate_system=CoordinateSystem.NORMALIZED,
                    )
                )
        return results

    def _parse_yolo_categories(self, data: str) -> list[SourceCategory]:
        return [
            SourceCategory(id=i, name=name.strip())
            for i, name in enumerate(data.strip().split("\n"))
            if name.strip()
        ]

    async def _parse_voc_annotations(self, data: str) -> list[SourceAnnotation]:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise AnnotationError(f"Invalid Pascal VOC XML: {e}")

        filename = root.findtext("filename", "")
        img_id = filename.replace(".xml", "")

        results: list[SourceAnnotation] = []
        for obj in root.findall("object"):
            cat = obj.findtext("name", "unknown")
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
                results.append(
                    SourceAnnotation(
                        id=f"{img_id}_{len(results)}",
                        image_id=img_id,
                        category_id=cat,
                        category_name=cat,
                        geometry_type=GeometryType.BBOX,
                        coordinates=(xmin, ymin, xmax - xmin, ymax - ymin),
                        coordinate_system=CoordinateSystem.PIXEL,
                    )
                )
        return results

    async def _parse_voc_categories(self, data: str) -> list[SourceCategory]:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise AnnotationError(f"Invalid Pascal VOC XML: {e}")
        seen: set[str] = set()
        categories: list[SourceCategory] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "unknown")
            if name not in seen:
                seen.add(name)
                categories.append(SourceCategory(id=name, name=name))
        return categories

    async def _parse_open_images_annotations(self, data: str) -> list[SourceAnnotation]:
        lines = [ln.strip() for ln in data.split("\n") if ln.strip()]
        results: list[SourceAnnotation] = []
        for i, line in enumerate(lines):
            if i == 0 and "ImageID" in line:
                continue
            parts = line.split(",")
            if len(parts) >= 8:
                x1, y1, x2, y2 = map(float, parts[4:8])
                results.append(
                    SourceAnnotation(
                        id=parts[0],
                        image_id=parts[1],
                        category_id=parts[2],
                        category_name=parts[2],
                        geometry_type=GeometryType.BBOX,
                        coordinates=(x1, y1, x2 - x1, y2 - y1),
                        coordinate_system=CoordinateSystem.PIXEL,
                        confidence=float(parts[3]) if parts[3] else None,
                    )
                )
        return results
