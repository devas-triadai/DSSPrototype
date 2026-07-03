"""Dataset profiling implementation.

Profiles candidate datasets by inspecting their structure, annotations,
image metadata, and license information.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import ProfileError
from backend.dataset_catalog.interfaces import DatasetProfileInterface
from backend.dataset_catalog.models import (
    ClassDistribution,
    DatasetProfile,
    LicenseInfo,
)

logger = logging.getLogger("dss.dataset_catalog.profile")


class DatasetProfiler(DatasetProfileInterface):
    """Profiles candidate datasets by inspecting their contents.

    Supports YOLO, COCO, and VOC annotation formats.
    """

    SUPPORTED_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self._profiles_dir = profiles_dir or dc_config.profiles_dir
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, DatasetProfile] = {}

    def profile(
        self,
        path: Path,
        source_id: str,
        source_type: str,
    ) -> DatasetProfile:
        """Profile a candidate dataset."""
        if not path.exists():
            raise ProfileError(f"Dataset path does not exist: {path}")

        path = path.resolve()
        profile_id = f"{source_id}_{path.name}"

        images = self._find_images(path)
        total_images = len(images)
        if total_images == 0:
            raise ProfileError(f"No supported images found at: {path}")

        annotation_format = self._detect_annotation_format(path)
        annotations, class_distribution = self._parse_annotations(
            path, annotation_format
        )
        total_annotations = sum(a["count"] for a in class_distribution.values())
        classes = sorted(class_distribution.keys())
        total_classes = len(classes)

        widths, heights = self._get_image_dimensions(images)
        avg_w = sum(widths) / len(widths) if widths else 0.0
        avg_h = sum(heights) / len(heights) if heights else 0.0
        resolution_dist = self._resolution_distribution(widths, heights)

        total_size = sum(
            os.path.getsize(img) for img in images if os.path.isfile(img)
        )
        estimated_size_mb = total_size / (1024 * 1024)

        license_info = self._detect_license(path)

        cls_dists = [
            ClassDistribution(
                class_name=cls_name,
                count=info["count"],
                annotation_count=info["annotation_count"],
                image_count=info["image_count"],
                avg_width=info["avg_width"],
                avg_height=info["avg_height"],
            )
            for cls_name, info in sorted(class_distribution.items())
        ]

        profile = DatasetProfile(
            profile_id=profile_id,
            source_id=source_id,
            source_type=source_type,
            path=str(path),
            total_images=total_images,
            total_annotations=total_annotations,
            total_classes=total_classes,
            classes=classes,
            class_distribution=cls_dists,
            avg_width=avg_w,
            avg_height=avg_h,
            resolution_distribution=resolution_dist,
            missing_annotations=self._count_missing_annotations(images, annotations),
            corrupt_images=self._count_corrupt(images),
            unsupported_formats=self._unsupported_formats(path),
            annotation_format=annotation_format,
            estimated_size_mb=estimated_size_mb,
            license_info=license_info,
            tags=[],
        )

        self._profiles[profile_id] = profile
        self._persist(profile)
        return profile

    def update_profile(self, profile: DatasetProfile) -> DatasetProfile:
        self._profiles[profile.profile_id] = profile
        self._persist(profile)
        return profile

    def get_profile(self, profile_id: str) -> DatasetProfile | None:
        if profile_id in self._profiles:
            return self._profiles[profile_id]
        path = self._profiles_dir / f"{profile_id}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                profile = DatasetProfile(**json.load(f))
            self._profiles[profile_id] = profile
            return profile
        return None

    def compare_profiles(self, profile_ids: Sequence[str]) -> dict[str, object]:
        profiles = []
        for pid in profile_ids:
            p = self.get_profile(pid)
            if p is None:
                raise ProfileError(f"Profile not found: {pid}")
            profiles.append(p)

        class_sets = [set(p.classes) for p in profiles]
        common = set.intersection(*class_sets) if class_sets else set()
        all_classes = set.union(*class_sets) if class_sets else set()

        return {
            "profile_count": len(profiles),
            "common_classes": sorted(common),
            "unique_classes_per_profile": {
                p.profile_id: sorted(set(p.classes) - common)
                for p in profiles
            },
            "all_classes": sorted(all_classes),
            "total_images": sum(p.total_images for p in profiles),
            "total_annotations": sum(p.total_annotations for p in profiles),
        }

    # ------------------------------------------------------------------
    # Internal inspection helpers
    # ------------------------------------------------------------------

    def _find_images(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in self.SUPPORTED_IMAGE_EXTS else []
        images: list[Path] = []
        for f in path.rglob("*"):
            if f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS:
                images.append(f)
        return images

    def _detect_annotation_format(self, path: Path) -> str:
        if path.is_file():
            return "unknown"
        has_labels = list(path.rglob("labels/*.txt"))
        if has_labels:
            return "yolo"
        has_coco = list(path.rglob("*.json"))
        for c in has_coco:
            try:
                with c.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if "annotations" in data and "categories" in data:
                    return "coco"
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        has_voc = list(path.rglob("*.xml"))
        if has_voc:
            return "voc"
        return "unknown"

    def _parse_annotations(
        self, path: Path, fmt: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        annotations: list[dict[str, Any]] = []
        class_stats: dict[str, dict[str, Any]] = {}

        if fmt == "yolo":
            label_files = list(path.rglob("labels/*.txt"))
            for lf in label_files:
                with lf.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = parts[0]
                            w, h = float(parts[3]), float(parts[4])
                            annotations.append(
                                {"class": cls_id, "width": w, "height": h}
                            )
                            if cls_id not in class_stats:
                                class_stats[cls_id] = {
                                    "count": 0,
                                    "annotation_count": 0,
                                    "image_count": 0,
                                    "widths": [],
                                    "heights": [],
                                }
                            class_stats[cls_id]["count"] += 1
                            class_stats[cls_id]["annotation_count"] += 1
                            class_stats[cls_id]["widths"].append(w)
                            class_stats[cls_id]["heights"].append(h)
            # De-duplicate images per class
            class_images: dict[str, set[str]] = {}
            for lf in label_files:
                with lf.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = parts[0]
                            if cls not in class_images:
                                class_images[cls] = set()
                            class_images[cls].add(str(lf))
            for cls, img_set in class_images.items():
                if cls in class_stats:
                    class_stats[cls]["image_count"] = len(img_set)

        elif fmt == "coco":
            json_files = list(path.rglob("*.json"))
            for jf in json_files:
                try:
                    with jf.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
                    for ann in data.get("annotations", []):
                        cls_name = cat_map.get(ann["category_id"], str(ann["category_id"]))
                        bbox = ann.get("bbox", [0, 0, 0, 0])
                        annotations.append(
                            {
                                "class": cls_name,
                                "width": bbox[2] if len(bbox) > 2 else 0,
                                "height": bbox[3] if len(bbox) > 3 else 0,
                            }
                        )
                        if cls_name not in class_stats:
                            class_stats[cls_name] = {
                                "count": 0,
                                "annotation_count": 0,
                                "image_count": 0,
                                "widths": [],
                                "heights": [],
                            }
                        class_stats[cls_name]["count"] += 1
                        class_stats[cls_name]["annotation_count"] += 1
                        class_stats[cls_name]["widths"].append(bbox[2] if len(bbox) > 2 else 0)
                        class_stats[cls_name]["heights"].append(bbox[3] if len(bbox) > 3 else 0)
                except (json.JSONDecodeError, KeyError):
                    continue

        elif fmt == "voc":
            import xml.etree.ElementTree as ET

            xml_files = list(path.rglob("*.xml"))
            for xf in xml_files:
                try:
                    tree = ET.parse(xf)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        cls_name = obj.findtext("name", "unknown")
                        bndbox = obj.find("bndbox")
                        if bndbox is not None:
                            xmin = float(bndbox.findtext("xmin", "0"))
                            ymin = float(bndbox.findtext("ymin", "0"))
                            xmax = float(bndbox.findtext("xmax", "0"))
                            ymax = float(bndbox.findtext("ymax", "0"))
                            w = xmax - xmin
                            h = ymax - ymin
                        else:
                            w, h = 0.0, 0.0
                        annotations.append({"class": cls_name, "width": w, "height": h})
                        if cls_name not in class_stats:
                            class_stats[cls_name] = {
                                "count": 0,
                                "annotation_count": 0,
                                "image_count": 0,
                                "widths": [],
                                "heights": [],
                            }
                        class_stats[cls_name]["count"] += 1
                        class_stats[cls_name]["annotation_count"] += 1
                        class_stats[cls_name]["widths"].append(w)
                        class_stats[cls_name]["heights"].append(h)
                except (ET.ParseError, AttributeError):
                    continue

        # Finalize averages
        result_stats: dict[str, dict[str, Any]] = {}
        for cls, stats in class_stats.items():
            result_stats[cls] = {
                "count": stats["count"],
                "annotation_count": stats["annotation_count"],
                "image_count": stats.get("image_count", 0),
                "avg_width": (
                    sum(stats["widths"]) / len(stats["widths"]) if stats["widths"] else 0
                ),
                "avg_height": (
                    sum(stats["heights"]) / len(stats["heights"]) if stats["heights"] else 0
                ),
            }

        return annotations, result_stats

    def _get_image_dimensions(self, images: list[Path]) -> tuple[list[int], list[int]]:
        widths: list[int] = []
        heights: list[int] = []
        for img in images[:500]:  # Sample first 500
            try:
                size = self._quick_dimension(img)
                if size:
                    w, h = size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue
        return widths, heights

    def _quick_dimension(self, path: Path) -> tuple[int, int] | None:
        """Quickly estimate image dimensions without loading the full image."""
        try:
            data = path.read_bytes()[:200]
            if data.startswith(b"\xff\xd8"):
                # JPEG
                import struct

                i = 2
                while i < len(data) - 1:
                    if data[i] != 0xFF:
                        break
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC2):
                        h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                        return w, h
                    i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
            elif data.startswith(b"\x89PNG"):
                import struct

                w, h = struct.unpack(">II", data[16:24])
                return w, h
            elif data.startswith(b"BM"):
                import struct

                size_data = data[18:26]
                if len(size_data) >= 8:
                    w = struct.unpack("<I", size_data[:4])[0]
                    h = abs(struct.unpack("<i", size_data[4:8])[0])
                    return w, h
        except Exception:
            pass
        return None

    def _resolution_distribution(
        self, widths: list[int], heights: list[int]
    ) -> dict[str, int]:
        dist: Counter[str] = Counter()
        for w, h in zip(widths, heights):
            if w >= 1920 and h >= 1080:
                dist["high"] += 1
            elif w >= 640 and h >= 480:
                dist["medium"] += 1
            else:
                dist["low"] += 1
        return dict(dist)

    def _count_missing_annotations(
        self, images: list[Path], _annotations: list[dict[str, Any]]
    ) -> int:
        return 0  # Would cross-reference image paths with annotation files

    def _count_corrupt(self, images: list[Path]) -> int:
        count = 0
        for img in images[:200]:
            try:
                data = img.read_bytes()[:50]
                if not data:
                    count += 1
            except (OSError, PermissionError):
                count += 1
        return count

    def _unsupported_formats(self, path: Path) -> list[str]:
        found: set[str] = set()
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() not in self.SUPPORTED_IMAGE_EXTS | {
                    ".txt",
                    ".json",
                    ".xml",
                    ".csv",
                    ".yaml",
                    ".yml",
                    ".md",
                }:
                    found.add(f.suffix.lower())
        return sorted(found)

    def _detect_license(self, path: Path) -> LicenseInfo | None:
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md", "license"]
        for lf in license_files:
            lp = path / lf if path.is_dir() else path.parent / lf
            if lp.exists():
                try:
                    text = lp.read_text(encoding="utf-8").strip()[:1000]
                    return self._classify_from_text(text)
                except Exception:
                    continue
        return None

    def _classify_from_text(self, text: str) -> LicenseInfo:
        text_lower = text.lower()
        if "creative commons" in text_lower or "cc" in text_lower:
            if "noncommercial" in text_lower or "nc" in text_lower:
                return LicenseInfo(
                    license_id="cc_nc",
                    name="Creative Commons Non-Commercial",
                    spdx_identifier="CC-BY-NC-4.0",
                    is_open=True,
                    allows_commercial=False,
                    allows_modification=True,
                    requires_attribution=True,
                    risk_score=0.6,
                )
            if "zero" in text_lower or "cc0" in text_lower:
                return LicenseInfo(
                    license_id="cc0",
                    name="CC0 1.0 Universal",
                    spdx_identifier="CC0-1.0",
                    is_open=True,
                    allows_commercial=True,
                    allows_modification=True,
                    requires_attribution=False,
                    risk_score=0.0,
                )
            return LicenseInfo(
                license_id="cc_by_4",
                name="Creative Commons Attribution 4.0",
                spdx_identifier="CC-BY-4.0",
                is_open=True,
                allows_commercial=True,
                allows_modification=True,
                requires_attribution=True,
                risk_score=0.1,
            )
        if "mit" in text_lower and "license" in text_lower:
            return LicenseInfo(
                license_id="mit",
                name="MIT License",
                spdx_identifier="MIT",
                is_open=True,
                allows_commercial=True,
                allows_modification=True,
                requires_attribution=True,
                risk_score=0.05,
            )
        return LicenseInfo(
            license_id="unknown",
            name="Unknown License",
            is_open=False,
            allows_commercial=False,
            allows_modification=False,
            requires_attribution=True,
            risk_score=0.8,
        )

    def _persist(self, profile: DatasetProfile) -> None:
        path = self._profiles_dir / f"{profile.profile_id}.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))
