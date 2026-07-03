"""GeoJSON file loader.

Converts GeoJSON ``FeatureCollection`` and ``Feature`` objects into
flat document dictionaries suitable for the knowledge base pipeline.
"""

import json
from pathlib import Path

from knowledge_base.loader.base import Loader, LoaderResult


class GeoJsonLoader(Loader):
    """Load terrain feature documents from a GeoJSON file.

    Supports both ``FeatureCollection`` (multiple features) and
    ``Feature`` (single feature) at the top level.  Each feature's
    ``properties`` dict is merged with a ``"geometry"`` key containing
    the GeoJSON geometry object.
    """

    def load(self, path: str) -> LoaderResult:
        source = str(Path(path).resolve())
        errors: list[str] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            return LoaderResult(documents=[], source=source, errors=[str(exc)])

        if isinstance(raw, dict):
            if raw.get("type") == "FeatureCollection":
                features = raw.get("features", [])
            elif raw.get("type") == "Feature":
                features = [raw]
            else:
                features = raw.get("documents", [])
        else:
            features = []

        if not isinstance(features, list):
            features = []
            errors.append(f"'features' is not a list in {source}")

        documents: list[dict[str, object]] = []
        for i, feature in enumerate(features):
            if not isinstance(feature, dict):
                errors.append(f"Feature at index {i} is not an object")
                continue
            properties = feature.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            doc: dict[str, object] = dict(properties)
            if "geometry" in feature:
                doc["geometry"] = feature["geometry"]
            documents.append(doc)

        return LoaderResult(
            documents=documents,
            source=source,
            count=len(documents),
            errors=errors,
        )
