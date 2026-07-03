"""Export ontology mappings to JSON, YAML, and CSV formats.

Each exporter produces a string representation suitable for
file I/O or API responses.
"""

from __future__ import annotations

import csv
import io
import json

from backend.ontology_mapping.exceptions import ExportError
from backend.ontology_mapping.models import DatasetMapping, ExportFormat


class MappingExporter:
    """Serializes dataset mappings to various formats."""

    async def to_json(self, mapping: DatasetMapping) -> str:
        try:
            data = mapping.model_dump(mode="python")
            data["created_at"] = data["created_at"].isoformat()
            data["updated_at"] = data["updated_at"].isoformat()
            rules_data = []
            for rule in data["rules"]:
                rule["created_at"] = rule["created_at"].isoformat()
                rules_data.append(rule)
            data["rules"] = rules_data
            return json.dumps(data, indent=2, default=str)
        except Exception as exc:
            raise ExportError(
                f"Failed to export to JSON: {exc}"
            ) from exc

    async def to_yaml(self, mapping: DatasetMapping) -> str:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ExportError(
                "PyYAML is not installed. Install with: pip install pyyaml"
            )

        try:
            data = mapping.model_dump(mode="python")
            data["created_at"] = data["created_at"].isoformat()
            data["updated_at"] = data["updated_at"].isoformat()
            rules_data = []
            for rule in data["rules"]:
                rule["created_at"] = rule["created_at"].isoformat()
                rules_data.append(rule)
            data["rules"] = rules_data
            return yaml.dump(
                data, default_flow_style=False, sort_keys=False
            )
        except Exception as exc:
            raise ExportError(
                f"Failed to export to YAML: {exc}"
            ) from exc

    async def to_csv(self, mapping: DatasetMapping) -> str:
        try:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "rule_id",
                "dataset_name",
                "source_label",
                "canonical_value",
                "match_type",
                "confidence",
                "version",
                "created_at",
            ])
            for rule in mapping.rules:
                writer.writerow([
                    rule.rule_id,
                    rule.dataset_name,
                    rule.source_label,
                    rule.canonical_value,
                    rule.match_type.value,
                    rule.confidence,
                    rule.version,
                    rule.created_at.isoformat(),
                ])
            return output.getvalue()
        except Exception as exc:
            raise ExportError(
                f"Failed to export to CSV: {exc}"
            ) from exc

    async def export_all(
        self,
        mapping: DatasetMapping,
        directory: str,
    ) -> dict[str, str]:
        from pathlib import Path

        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        json_path = base / f"{mapping.dataset_name}_mapping.json"
        yaml_path = base / f"{mapping.dataset_name}_mapping.yaml"
        csv_path = base / f"{mapping.dataset_name}_mapping.csv"

        json_content = await self.to_json(mapping)
        yaml_content = await self.to_yaml(mapping)
        csv_content = await self.to_csv(mapping)

        json_path.write_text(json_content, encoding="utf-8")
        yaml_path.write_text(yaml_content, encoding="utf-8")
        csv_path.write_text(csv_content, encoding="utf-8")

        return {
            ExportFormat.JSON: str(json_path),
            ExportFormat.YAML: str(yaml_path),
            ExportFormat.CSV: str(csv_path),
        }

    async def export(
        self,
        mapping: DatasetMapping,
        fmt: ExportFormat,
    ) -> str:
        if fmt == ExportFormat.JSON:
            return await self.to_json(mapping)
        if fmt == ExportFormat.YAML:
            return await self.to_yaml(mapping)
        if fmt == ExportFormat.CSV:
            return await self.to_csv(mapping)
        raise ExportError(f"Unsupported export format: {fmt}")
