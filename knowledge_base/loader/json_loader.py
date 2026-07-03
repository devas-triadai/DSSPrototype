"""JSON file loader."""

import json
from pathlib import Path

from knowledge_base.loader.base import Loader, LoaderResult


class JsonLoader(Loader):
    """Load knowledge documents from a JSON file.

    The file must contain either a JSON array of objects or a
    JSON object with a ``"documents"`` key containing the array.
    """

    def load(self, path: str) -> LoaderResult:
        source = str(Path(path).resolve())
        errors: list[str] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            return LoaderResult(documents=[], source=source, errors=[str(exc)])

        if isinstance(raw, list):
            documents = raw
        elif isinstance(raw, dict):
            documents = raw.get("documents", [])
        else:
            errors.append(f"Unexpected JSON structure at {source}")

        if not isinstance(documents, list):
            documents = []
            errors.append(f"'documents' key is not a list in {source}")

        return LoaderResult(
            documents=documents,
            source=source,
            count=len(documents),
            errors=errors,
        )
