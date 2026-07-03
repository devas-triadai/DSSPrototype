"""Curation workflow management for dataset review and approval."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import (
    CurationError,
    CurationWorkflowError,
    EntryNotFoundError,
)
from backend.dataset_catalog.interfaces import (
    CatalogInterface,
    CurationServiceInterface,
)
from backend.dataset_catalog.models import (
    CatalogEntry,
    CurationRecord,
)

logger = logging.getLogger("dss.dataset_catalog.curation")


class CurationService(CurationServiceInterface):
    """Manages the curation workflow for dataset catalog entries.

    Workflow: draft → pending_review → approved | rejected
    """

    def __init__(
        self,
        catalog: CatalogInterface,
        work_dir: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._work_dir = work_dir or dc_config.work_dir
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, CurationRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        records_file = self._work_dir / "curation_records.json"
        if records_file.exists():
            try:
                with records_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._records = {
                    rid: CurationRecord(**data) for rid, data in raw.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load curation records: %s", exc)

    def _save(self) -> None:
        records_file = self._work_dir / "curation_records.json"
        raw = {
            rid: rec.model_dump(mode="json")
            for rid, rec in self._records.items()
        }
        with records_file.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def create_record(self, entry_id: str, curator: str) -> CurationRecord:
        entry = self._catalog.get_entry(entry_id)
        if entry is None:
            raise EntryNotFoundError(f"Entry not found: {entry_id}")

        record_id = f"cur_{entry_id}_{curator}"
        with self._lock:
            record = CurationRecord(
                record_id=record_id,
                entry_id=entry_id,
                curator=curator,
                status="draft",
            )
            self._records[record_id] = record
            self._save()
            return record

    def submit_for_review(self, record_id: str) -> CurationRecord:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise CurationError(f"Record not found: {record_id}")
            if record.status != "draft":
                raise CurationWorkflowError(
                    f"Cannot submit record {record_id}: "
                    f"current status is '{record.status}', expected 'draft'"
                )

            # Check pending review limit
            pending = sum(
                1 for r in self._records.values() if r.status == "pending_review"
            )
            if pending >= dc_config.max_pending_review_items:
                raise CurationWorkflowError(
                    f"Max pending review items reached ({dc_config.max_pending_review_items})"
                )

            updated = CurationRecord(
                record_id=record.record_id,
                entry_id=record.entry_id,
                curator=record.curator,
                status="pending_review",
                reviewer=record.reviewer,
                review_notes=record.review_notes,
                rejection_reason=record.rejection_reason,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record_id] = updated
            self._save()
            return updated

    def approve(self, record_id: str, reviewer: str) -> CurationRecord:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise CurationError(f"Record not found: {record_id}")
            if record.status != "pending_review":
                raise CurationWorkflowError(
                    f"Cannot approve record {record_id}: "
                    f"current status is '{record.status}', expected 'pending_review'"
                )

            updated = CurationRecord(
                record_id=record.record_id,
                entry_id=record.entry_id,
                curator=record.curator,
                status="approved",
                reviewer=reviewer,
                review_notes=record.review_notes,
                rejection_reason="",
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
                approved_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record_id] = updated
            self._save()

            # Update catalog entry status
            entry = self._catalog.get_entry(record.entry_id)
            if entry:
                updated_entry = CatalogEntry(
                    entry_id=entry.entry_id,
                    name=entry.name,
                    source_id=entry.source_id,
                    source_type=entry.source_type,
                    domain=entry.domain,
                    status="acquired",
                    profile=entry.profile,
                    quality_score=entry.quality_score,
                    coverage_score=entry.coverage_score,
                    diversity_score=entry.diversity_score,
                    license_score=entry.license_score,
                    overall_score=entry.overall_score,
                    tags=entry.tags,
                    notes=entry.notes,
                    created_at=entry.created_at,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self._catalog.update_entry(updated_entry)

            return updated

    def reject(
        self, record_id: str, reviewer: str, reason: str
    ) -> CurationRecord:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                raise CurationError(f"Record not found: {record_id}")
            if record.status != "pending_review":
                raise CurationWorkflowError(
                    f"Cannot reject record {record_id}: "
                    f"current status is '{record.status}', expected 'pending_review'"
                )

            updated = CurationRecord(
                record_id=record.record_id,
                entry_id=record.entry_id,
                curator=record.curator,
                status="rejected",
                reviewer=reviewer,
                review_notes=record.review_notes,
                rejection_reason=reason,
                created_at=record.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
                rejected_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record_id] = updated
            self._save()
            return updated

    def get_record(self, record_id: str) -> CurationRecord | None:
        return self._records.get(record_id)

    def list_pending(self) -> list[CurationRecord]:
        return [
            r
            for r in self._records.values()
            if r.status == "pending_review"
        ]

    def list_by_curator(self, curator: str) -> list[CurationRecord]:
        return [
            r
            for r in self._records.values()
            if r.curator == curator
        ]
