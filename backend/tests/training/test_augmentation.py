"""Tests for the AugmentationPipeline."""

from backend.training.augmentation import AugmentationPipeline, TransformDescriptor
from backend.training.models import AugmentationConfig


def _make_pipeline() -> AugmentationPipeline:
    return AugmentationPipeline()


def _get_transforms(result: object) -> list[dict[str, object]]:
    assert isinstance(result, dict)
    trans = result.get("transforms", [])
    assert isinstance(trans, list)
    result_list: list[dict[str, object]] = []
    for t in trans:
        assert isinstance(t, dict)
        result_list.append(t)
    return result_list


def test_create_pipeline_default() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig()
    result = pipeline.create_pipeline(config)
    assert isinstance(result, dict)
    assert result["name"] == "default"
    assert len(_get_transforms(result)) == 10


def test_create_pipeline_with_flip_only() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(
        flip_probability=0.5,
        rotate_probability=0.0,
        scale_probability=0.0,
        crop_probability=0.0,
        brightness_probability=0.0,
        contrast_probability=0.0,
        blur_probability=0.0,
        noise_probability=0.0,
        mosaic_probability=0.0,
        mixup_probability=0.0,
    )
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    assert len(transforms) == 1
    assert transforms[0]["type"] == "flip"


def test_create_pipeline_all_disabled() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(
        flip_probability=0.0,
        rotate_probability=0.0,
        scale_probability=0.0,
        crop_probability=0.0,
        brightness_probability=0.0,
        contrast_probability=0.0,
        blur_probability=0.0,
        noise_probability=0.0,
        mosaic_probability=0.0,
        mixup_probability=0.0,
    )
    result = pipeline.create_pipeline(config)
    assert _get_transforms(result) == []


def test_create_pipeline_partial() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(
        flip_probability=0.5,
        rotate_probability=0.5,
        scale_probability=0.0,
        crop_probability=0.0,
        brightness_probability=0.0,
        contrast_probability=0.0,
        blur_probability=0.0,
        noise_probability=0.0,
        mosaic_probability=0.0,
        mixup_probability=0.0,
    )
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    assert len(transforms) == 2
    assert transforms[0]["type"] == "flip"
    assert transforms[1]["type"] == "rotate"


def test_rotate_transform_has_degrees() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(rotate_probability=0.5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    rotate = [t for t in transforms if t["type"] == "rotate"][0]
    assert rotate["degrees"] == [-10.0, 10.0]


def test_scale_transform_has_min_max() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(scale_probability=0.5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    scale = [t for t in transforms if t["type"] == "scale"][0]
    assert scale["min"] == 0.5
    assert scale["max"] == 2.0


def test_blur_transform_has_kernel_size() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(blur_probability=0.5, blur_kernel_size=5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    blur = [t for t in transforms if t["type"] == "blur"][0]
    assert blur["kernel_size"] == 5


def test_brightness_transform_has_range() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(brightness_probability=0.5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    bri = [t for t in transforms if t["type"] == "brightness"][0]
    assert bri["range"] == [0.8, 1.2]


def test_mosaic_transform() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(mosaic_probability=0.5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    mosaic = [t for t in transforms if t["type"] == "mosaic"]
    assert len(mosaic) == 1
    assert mosaic[0]["probability"] == 0.5


def test_mixup_transform() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(mixup_probability=0.5)
    result = pipeline.create_pipeline(config)
    transforms = _get_transforms(result)
    mixup = [t for t in transforms if t["type"] == "mixup"]
    assert len(mixup) == 1


def test_get_available_transforms() -> None:
    pipeline = _make_pipeline()
    transforms = pipeline.get_available_transforms()
    assert "flip" in transforms
    assert "rotate" in transforms
    assert "scale" in transforms
    assert "crop" in transforms
    assert "brightness" in transforms
    assert "contrast" in transforms
    assert "blur" in transforms
    assert "noise" in transforms
    assert "mosaic" in transforms
    assert "mixup" in transforms
    assert len(transforms) == 10


def test_transform_descriptor() -> None:
    td = TransformDescriptor("test", "Test transform", ("param1", "param2"))
    assert td.name == "test"
    assert td.description == "Test transform"
    assert td.configurable_params == ("param1", "param2")


def test_create_pipeline_preserves_name() -> None:
    pipeline = _make_pipeline()
    config = AugmentationConfig(name="my_aug")
    result = pipeline.create_pipeline(config)
    assert isinstance(result, dict)
    assert result["name"] == "my_aug"
