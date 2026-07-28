"""Public entry point for the Computer Vision module.

Coordinates the full detection pipeline while containing zero
image-processing or model-specific logic.
"""

from backend.contracts.enums.core import ObjectType
from backend.contracts.interfaces.vision import VisionModule
from backend.contracts.models.detection import DetectionResult, ImageMetadata
from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.exceptions import ModelLoadError
from backend.modules.computer_vision.image_loader import ImageLoader
from backend.modules.computer_vision.image_preprocessor import ImagePreprocessor
from backend.modules.computer_vision.image_validator import ImageValidator
from backend.modules.computer_vision.inference_engine import InferenceEngine
from backend.modules.computer_vision.interfaces import (
    ImageLoaderInterface,
    ImagePreprocessorInterface,
    ImageValidatorInterface,
    InferenceEngineInterface,
    ModelManagerInterface,
    ResultConverterInterface,
)
from backend.modules.computer_vision.model_manager import ModelManager
from backend.modules.computer_vision.result_converter import ResultConverter

# Maps COCO class IDs (0-79) to DSS domain ObjectType
# Only perception-relevant classes are mapped. All others → UNKNOWN_OBJECT.
_YOLO_TO_OBJECT_TYPE: dict[int, ObjectType] = {
    0: ObjectType.PEOPLE_PERSON,
    1: ObjectType.GROUND_VEHICLE_BICYCLE,
    2: ObjectType.GROUND_VEHICLE_CAR,
    3: ObjectType.GROUND_VEHICLE_MOTORCYCLE,
    4: ObjectType.AIRCRAFT_FIXED_WING,
    5: ObjectType.GROUND_VEHICLE_BUS,
    6: ObjectType.RAIL_TRAIN,
    7: ObjectType.GROUND_VEHICLE_TRUCK,
    8: ObjectType.WATERCRAFT_BOAT,
    9: ObjectType.INFRASTRUCTURE_TRAFFIC_LIGHT,
    11: ObjectType.ROAD_NETWORK_ROAD_SIGN,
    13: ObjectType.INFRASTRUCTURE_BENCH,
}  # Remaining COCO classes (indoor objects, animals, food) → UNKNOWN_OBJECT

# Name-based fallback for custom models (e.g. military YOLO).
# Keys are normalised: lowercase, underscores for spaces/hyphens.
_MILITARY_NAME_TO_OBJECT_TYPE: dict[str, ObjectType] = {
    "artillery": ObjectType.MILITARY_ARTILLERY,
    "missile": ObjectType.MILITARY_MISSILE,
    "tank": ObjectType.MILITARY_TANK,
    "armored_vehicle": ObjectType.MILITARY_ARMORED_VEHICLE,
    "baktar_shikan_atgm": ObjectType.MILITARY_MISSILE,
    "atgm": ObjectType.MILITARY_MISSILE,
    "anti_tank_guided_missile": ObjectType.MILITARY_MISSILE,
    "howitzer": ObjectType.MILITARY_ARTILLERY,
    "rocket_launcher": ObjectType.MILITARY_ARTILLERY,
    "mlrs": ObjectType.MILITARY_ARTILLERY,
    "bmp": ObjectType.MILITARY_ARMORED_VEHICLE,
    "apc": ObjectType.MILITARY_ARMORED_VEHICLE,
    "ifv": ObjectType.MILITARY_ARMORED_VEHICLE,
    "person": ObjectType.PEOPLE_PERSON,
    "soldier": ObjectType.PEOPLE_PERSON,
    "car": ObjectType.GROUND_VEHICLE_CAR,
    "truck": ObjectType.GROUND_VEHICLE_TRUCK,
    "bus": ObjectType.GROUND_VEHICLE_BUS,
    "aircraft": ObjectType.AIRCRAFT_FIXED_WING,
    "helicopter": ObjectType.AIRCRAFT_ROTARY_WING,
    "drone": ObjectType.AIRCRAFT_UAV,
    "ship": ObjectType.WATERCRAFT_SHIP,
    "boat": ObjectType.WATERCRAFT_BOAT,
    "building": ObjectType.BUILDINGS_BUILDING,
    "bridge": ObjectType.BRIDGES_BEAM,
    "road": ObjectType.ROAD_NETWORK_ROAD,
}


class ComputerVisionService(VisionModule):
    """Orchestrates the end-to-end computer vision pipeline.

    Pipeline steps:
        1. Load image           (``ImageLoaderInterface``)
        2. Validate image       (``ImageValidatorInterface``)
        3. Preprocess image     (``ImagePreprocessorInterface``)
        4. Acquire model        (``ModelManagerInterface``)
        5. Run inference        (``InferenceEngineInterface``)
        6. Convert results      (``ResultConverterInterface``)

    Dependencies are injected via the constructor; sensible defaults
    are provided for every component.
    """

    def __init__(
        self,
        image_loader: ImageLoaderInterface | None = None,
        image_validator: ImageValidatorInterface | None = None,
        image_preprocessor: ImagePreprocessorInterface | None = None,
        model_manager: ModelManagerInterface | None = None,
        inference_engine: InferenceEngineInterface | None = None,
        result_converter: ResultConverterInterface | None = None,
    ) -> None:
        self._image_loader = image_loader or ImageLoader()
        self._image_validator = image_validator or ImageValidator()
        self._image_preprocessor = image_preprocessor or ImagePreprocessor()
        self._model_manager = model_manager or ModelManager()
        self._inference_engine = inference_engine or InferenceEngine()
        self._result_converter = result_converter or ResultConverter(
            class_mapping=_YOLO_TO_OBJECT_TYPE,
            class_name_mapping=_MILITARY_NAME_TO_OBJECT_TYPE,
        )
        self._config = cv_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_image(self, image_meta: ImageMetadata) -> DetectionResult:
        """Run the full detection pipeline on a single image.

        The image data is loaded using *image_meta.image_id* as a
        filesystem path or identifier.  In production this would
        be resolved via the upload folder configured in settings.

        Parameters
        ----------
        image_meta:
            Metadata describing the image to process.

        Returns
        -------
        DetectionResult
            Strongly typed detection output.
        """
        image_path = image_meta.image_id
        source: str | bytes

        source = image_path

        image_array = self._image_loader.load(source)
        self._image_validator.validate(image_array)
        processed = self._image_preprocessor.preprocess(image_array)

        try:
            model = self._model_manager.get_model(self._config.default_model_type)
        except ModelLoadError:
            model = self._model_manager.load_model(
                self._config.default_model_type,
                self._config.model_path,
            )
        raw_output = self._inference_engine.run(model, processed)

        return self._result_converter.convert(
            raw_output,
            image_meta,
            model_version=model.metadata.version,
        )
