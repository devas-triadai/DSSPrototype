"""Data loaders for the DSS Knowledge Base.

Supports JSON, CSV, YAML, and GeoJSON input formats.  Each loader
returns a list of validated document dictionaries ready for indexing.
"""

from knowledge_base.loader.base import Loader, LoaderResult
from knowledge_base.loader.csv_loader import CsvLoader
from knowledge_base.loader.geojson_loader import GeoJsonLoader
from knowledge_base.loader.json_loader import JsonLoader
from knowledge_base.loader.yaml_loader import YamlLoader

__all__ = [
    "Loader",
    "LoaderResult",
    "JsonLoader",
    "CsvLoader",
    "YamlLoader",
    "GeoJsonLoader",
]
