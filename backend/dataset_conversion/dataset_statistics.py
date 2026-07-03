from __future__ import annotations

from collections import defaultdict

from backend.dataset_conversion.models import CanonicalDataset, DatasetStatistics


class DatasetStatisticsGenerator:
    async def compute(
        self,
        dataset: CanonicalDataset,
    ) -> DatasetStatistics:
        images = dataset.images
        annotations = dataset.annotations

        total_images = len(images)
        total_annotations = len(annotations)

        anns_per_image: dict[str, int] = defaultdict(int)
        class_counts: dict[str, int] = defaultdict(int)
        total_bbox_w = 0.0
        total_bbox_h = 0.0
        total_img_w = 0.0
        total_img_h = 0.0
        img_w_count = 0
        img_h_count = 0

        for ann in annotations:
            anns_per_image[ann.image_id] += 1
            class_counts[ann.canonical_label] += 1
            total_bbox_w += ann.width
            total_bbox_h += ann.height

        for img in images:
            if img.width:
                total_img_w += img.width
                img_w_count += 1
            if img.height:
                total_img_h += img.height
                img_h_count += 1

        images_with_anns = sum(1 for img_id in anns_per_image if anns_per_image[img_id] > 0)
        images_without_anns = total_images - images_with_anns

        avg_anns = total_annotations / total_images if total_images > 0 else 0.0
        min_anns = min(anns_per_image.values()) if anns_per_image else 0
        max_anns = max(anns_per_image.values()) if anns_per_image else 0
        avg_img_w = total_img_w / img_w_count if img_w_count > 0 else 0.0
        avg_img_h = total_img_h / img_h_count if img_h_count > 0 else 0.0
        avg_bbox_w = total_bbox_w / total_annotations if total_annotations > 0 else 0.0
        avg_bbox_h = total_bbox_h / total_annotations if total_annotations > 0 else 0.0

        class_balance: dict[str, float] = {}
        for cls_name, count in class_counts.items():
            class_balance[cls_name] = count / total_annotations if total_annotations > 0 else 0.0

        total_classes = len(class_counts)

        coverage: dict[str, object] = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "total_classes": total_classes,
            "images_with_annotations": images_with_anns,
            "images_without_annotations": images_without_anns,
            "coverage_pct": (images_with_anns / total_images * 100) if total_images > 0 else 0.0,
        }

        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])

        return DatasetStatistics(
            total_images=total_images,
            total_annotations=total_annotations,
            total_classes=total_classes,
            classes=tuple(sorted_classes),
            images_with_annotations=images_with_anns,
            images_without_annotations=images_without_anns,
            avg_annotations_per_image=avg_anns,
            min_annotations_per_image=min_anns,
            max_annotations_per_image=max_anns,
            avg_image_width=avg_img_w,
            avg_image_height=avg_img_h,
            avg_bbox_width=avg_bbox_w,
            avg_bbox_height=avg_bbox_h,
            class_balance=class_balance,
            coverage_report=coverage,
        )
