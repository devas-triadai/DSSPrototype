"""YAML file loader."""

from pathlib import Path

from knowledge_base.loader.base import Loader, LoaderResult


class YamlLoader(Loader):
    """Load knowledge documents from a YAML file.

    The file must contain either a YAML sequence of objects or a
    YAML mapping with a ``"documents"`` key containing the sequence.
    """

    def load(self, path: str) -> LoaderResult:
        source = str(Path(path).resolve())
        errors: list[str] = []
        try:
            import yaml
        except ImportError:
            return LoaderResult(
                documents=[],
                source=source,
                errors=["PyYAML is not installed — install with 'pip install pyyaml'"],
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            return LoaderResult(documents=[], source=source, errors=[str(exc)])

        if raw is None:
            raw = []

        if isinstance(raw, list):
            documents = raw
        elif isinstance(raw, dict):
            documents = raw.get("documents", [])
        else:
            documents = []
            errors.append(f"Unexpected YAML structure at {source}")

        if not isinstance(documents, list):
            documents = []
            errors.append(f"'documents' key is not a list in {source}")

        return LoaderResult(
            documents=documents,
            source=source,
            count=len(documents),
            errors=errors,
        )
