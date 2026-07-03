"""CSV file loader."""

import csv
from pathlib import Path

from knowledge_base.loader.base import Loader, LoaderResult


class CsvLoader(Loader):
    """Load knowledge documents from a CSV file.

    The first row must be a header line.  Each subsequent row is
    converted to a dictionary keyed by the header names.
    """

    def load(self, path: str) -> LoaderResult:
        source = str(Path(path).resolve())
        errors: list[str] = []
        documents: list[dict[str, str]] = []
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    documents.append(dict(row))
        except Exception as exc:
            return LoaderResult(documents=[], source=source, errors=[str(exc)])

        return LoaderResult(
            documents=documents,
            source=source,
            count=len(documents),
            errors=errors,
        )
