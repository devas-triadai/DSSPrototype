"""ConversionRequestBuilder — translates DatasetLayout into load_dataset() inputs.

Each dataset type (COCO, OpenImages V7, VisDrone, etc.) has a dedicated
builder method with knowledge of its annotation file structures, image
directory conventions, and the correct loader format string.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from backend.dataset_pipeline.models import DatasetLayout


class ConversionRequest(BaseModel):
    """Structured inputs for ``DatasetConversionService.load_dataset()``.

    Attributes:
        source_path:  File or directory path expected by the loader.
        dataset_format:  Format string (e.g. ``"coco_json"``, ``"open_images_csv"``).
        kwargs:  Additional keyword arguments forwarded to the loader
            (e.g. ``data_dir``, ``image_dir``, ``class_names_path``).
    """

    source_path: str
    dataset_format: str
    kwargs: dict[str, str] = Field(default_factory=dict)


class ConversionRequestBuilder:
    """Translates ``DatasetLayout`` into a ``ConversionRequest``.

    Each dataset type that the pipeline recognises gets its own builder
    method.  Unknown types raise ``ValueError``.
    """

    def build(self, layout: DatasetLayout) -> ConversionRequest:
        """Produce a ``ConversionRequest`` from a detected layout."""
        builder = self._get_builder(layout.dataset_type)
        return builder(layout)

    def _get_builder(self, dataset_type: str) -> Callable[[DatasetLayout], ConversionRequest]:
        builders = {
            "coco": self._build_coco,
            "open_images_v7": self._build_open_images_v7,
            "visdrone": self._build_visdrone,
            "loveda": self._build_loveda,
            "spacenet": self._build_spacenet,
            "seaships": self._build_seaships,
        }
        builder = builders.get(dataset_type)
        if builder is None:
            raise ValueError(
                f"Unsupported dataset type: {dataset_type!r}. "
                f"Supported types: {', '.join(sorted(builders))}"
            )
        return builder

    # ------------------------------------------------------------------
    # Per-type builders
    # ------------------------------------------------------------------

    def _build_coco(self, layout: DatasetLayout) -> ConversionRequest:
        """COCO JSON — annotation file is `*.json`, images are in a sibling directory."""
        ann_file = (
            layout.annotation_files[0]
            if layout.annotation_files
            else layout.root_path / "annotations" / "instances_train2017.json"
        )
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "train2017")
        )
        return ConversionRequest(
            source_path=str(ann_file),
            dataset_format="coco_json",
            kwargs={"data_dir": data_dir},
        )

    def _build_open_images_v7(self, layout: DatasetLayout) -> ConversionRequest:
        """Open Images V7 — annotation CSV at root, images in sub-directory."""
        csv_path = layout.root_path / "train-annotations-bbox.csv"
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "train")
        )
        return ConversionRequest(
            source_path=str(csv_path),
            dataset_format="open_images_csv",
            kwargs={"data_dir": data_dir},
        )

    def _build_visdrone(self, layout: DatasetLayout) -> ConversionRequest:
        """VisDrone — TXT annotations in *annotations/*, images in *images/*."""
        ann_dir = (
            layout.annotation_directory
            or layout.root_path / "VisDrone2019-DET-train" / "annotations"
        )
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "VisDrone2019-DET-train" / "images")
        )
        return ConversionRequest(
            source_path=str(ann_dir),
            dataset_format="coco_json",
            kwargs={"data_dir": data_dir},
        )

    def _build_loveda(self, layout: DatasetLayout) -> ConversionRequest:
        """LoveDA — GeoTIFF + GeoJSON; pass root directory with a data_dir hint."""
        source = (
            str(layout.annotation_files[0]) if layout.annotation_files else str(layout.root_path)
        )
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "Train")
        )
        return ConversionRequest(
            source_path=source,
            dataset_format="coco_json",
            kwargs={"data_dir": data_dir},
        )

    def _build_spacenet(self, layout: DatasetLayout) -> ConversionRequest:
        """SpaceNet — GeoJSON annotations; pass root directory with a data_dir hint."""
        source = (
            str(layout.annotation_files[0]) if layout.annotation_files else str(layout.root_path)
        )
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "AOI_2_Vegas")
        )
        return ConversionRequest(
            source_path=source,
            dataset_format="coco_json",
            kwargs={"data_dir": data_dir},
        )

    def _build_seaships(self, layout: DatasetLayout) -> ConversionRequest:
        """SeaShips — XML annotations; pass root with a data_dir hint."""
        source = (
            str(layout.annotation_files[0]) if layout.annotation_files else str(layout.root_path)
        )
        data_dir = (
            str(layout.image_directories[0])
            if layout.image_directories
            else str(layout.root_path / "train")
        )
        return ConversionRequest(
            source_path=source,
            dataset_format="coco_json",
            kwargs={"data_dir": data_dir},
        )
