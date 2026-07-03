"""Dataset loader — discovers images and annotations on disk.

Supports recursive loading through subdirectories with configurable
extension filtering.
"""

import logging
from pathlib import Path

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import DatasetLoaderInterface

logger = logging.getLogger("dss.dataset_manager.loader")


class DatasetLoader(DatasetLoaderInterface):
    """Loads dataset paths from disk.

    Discovers images and annotations recursively, filtering by
    supported extensions defined in DatasetManagerConfig.
    """

    def __init__(self) -> None:
        self._config = dm_config

    def load_images(self, path: Path) -> list[Path]:
        logger.debug("Loading images from: %s", path)
        exts = self._config.supported_image_extensions
        images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]
        logger.info("Loaded %d images from %s", len(images), path)
        return images

    def load_annotations(self, path: Path) -> list[Path]:
        logger.debug("Loading annotations from: %s", path)
        exts = self._config.supported_annotation_extensions
        annotations = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]
        logger.info("Loaded %d annotations from %s", len(annotations), path)
        return annotations
