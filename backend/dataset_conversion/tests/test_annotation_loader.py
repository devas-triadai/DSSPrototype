from __future__ import annotations

import json

import pytest

from backend.dataset_conversion.annotation_loader import AnnotationLoader
from backend.dataset_conversion.models import GeometryType


class TestAnnotationLoader:
    @pytest.fixture
    def loader(self) -> AnnotationLoader:
        return AnnotationLoader()

    @pytest.mark.asyncio
    async def test_parse_coco_annotations(
        self,
        loader: AnnotationLoader,
        coco_json_data: str,
    ) -> None:
        result = await loader.parse_annotations(coco_json_data, "coco_json")
        assert len(result) == 1
        assert result[0].category_name == "car"
        assert result[0].coordinates == (10.0, 20.0, 100.0, 200.0)

    @pytest.mark.asyncio
    async def test_parse_coco_categories(
        self,
        loader: AnnotationLoader,
        coco_json_data: str,
    ) -> None:
        result = await loader.parse_categories(coco_json_data, "coco_json")
        assert len(result) == 1
        assert result[0].name == "car"
        assert result[0].supercategory == "vehicle"

    @pytest.mark.asyncio
    async def test_parse_yolo_annotations(
        self,
        loader: AnnotationLoader,
        yolo_txt_data: str,
    ) -> None:
        result = await loader.parse_annotations(yolo_txt_data, "yolo_txt")
        assert len(result) == 2
        assert result[0].geometry_type == GeometryType.NORMALIZED
        assert result[0].category_id == 0

    @pytest.mark.asyncio
    async def test_parse_yolo_categories(
        self,
        loader: AnnotationLoader,
    ) -> None:
        data = "car\nperson\ntruck\n"
        result = loader._parse_yolo_categories(data)
        assert len(result) == 3
        assert result[2].name == "truck"

    @pytest.mark.asyncio
    async def test_parse_voc_annotations(
        self,
        loader: AnnotationLoader,
        pascal_voc_xml_data: str,
    ) -> None:
        result = await loader.parse_annotations(pascal_voc_xml_data, "pascal_voc")
        assert len(result) == 2
        assert result[0].category_name == "car"
        assert result[1].category_name == "person"

    @pytest.mark.asyncio
    async def test_parse_voc_categories(
        self,
        loader: AnnotationLoader,
        pascal_voc_xml_data: str,
    ) -> None:
        result = await loader.parse_categories(pascal_voc_xml_data, "pascal_voc")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parse_open_images(
        self,
        loader: AnnotationLoader,
        open_images_csv_data: str,
    ) -> None:
        result = await loader.parse_annotations(open_images_csv_data, "open_images_csv")
        assert len(result) == 2
        assert result[0].category_name == "Car"

    @pytest.mark.asyncio
    async def test_unsupported_format(
        self,
        loader: AnnotationLoader,
    ) -> None:
        with pytest.raises(Exception):
            await loader.parse_annotations("data", "unknown_format")

    @pytest.mark.asyncio
    async def test_coco_invalid_json(
        self,
        loader: AnnotationLoader,
    ) -> None:
        with pytest.raises(Exception):
            await loader.parse_annotations("not json", "coco_json")

    @pytest.mark.asyncio
    async def test_voc_no_objects(
        self,
        loader: AnnotationLoader,
    ) -> None:
        xml = "<annotation><filename>test.jpg</filename></annotation>"
        result = await loader.parse_annotations(xml, "pascal_voc")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_yolo_annotations_insufficient_coords(
        self,
        loader: AnnotationLoader,
    ) -> None:
        result = await loader.parse_annotations("0 0.5 0.5\n", "yolo_txt")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_yolo_annotations_many_lines(
        self,
        loader: AnnotationLoader,
    ) -> None:
        data = "\n".join(f"{i % 5} 0.5 0.5 0.3 0.4" for i in range(20))
        result = await loader.parse_annotations(data, "yolo_txt")
        assert len(result) == 20

    @pytest.mark.asyncio
    async def test_coco_with_segmentation(
        self,
        loader: AnnotationLoader,
    ) -> None:
        data = json.dumps(
            {
                "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    }
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        )
        result = await loader.parse_annotations(data, "coco_json")
        assert result[0].geometry_type == GeometryType.SEGMENTATION

    @pytest.mark.asyncio
    async def test_coco_no_categories(
        self,
        loader: AnnotationLoader,
    ) -> None:
        data = '{"images": [], "annotations": [], "categories": []}'
        result = await loader.parse_categories(data, "coco_json")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_voc_empty_xml(self, loader: AnnotationLoader) -> None:
        result = await loader.parse_annotations("<annotation></annotation>", "pascal_voc")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_yolo_empty_lines(self, loader: AnnotationLoader) -> None:
        result = await loader.parse_annotations("\n\n\n", "yolo_txt")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_yolo_varying_coordinate_count(self, loader: AnnotationLoader) -> None:
        result = await loader.parse_annotations("0 0.5\n1 0.2 0.3 0.1 0.2 0.9\n", "yolo_txt")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_open_images_skip_header(
        self,
        loader: AnnotationLoader,
    ) -> None:
        hdr = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax"
        data = f"{hdr}\nimg001,freeform,Car,1.0,0.1,0.5,0.2,0.6\n"
        result = await loader.parse_annotations(data, "open_images_csv")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_open_images_insufficient_columns(
        self,
        loader: AnnotationLoader,
    ) -> None:
        data = "img001,car\n"
        result = await loader.parse_annotations(data, "open_images_csv")
        assert len(result) == 0
