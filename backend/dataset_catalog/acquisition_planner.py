"""Acquisition planning for prioritized dataset acquisition."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import (
    AcquisitionError,
    AcquisitionLimitError,
    EntryNotFoundError,
)
from backend.dataset_catalog.interfaces import (
    AcquisitionPlannerInterface,
    CatalogInterface,
)
from backend.dataset_catalog.models import AcquisitionPlan

logger = logging.getLogger("dss.dataset_catalog.acquisition_planner")


class AcquisitionPlanner(AcquisitionPlannerInterface):
    """Manages dataset acquisition plans with priority scoring and limits."""

    def __init__(
        self,
        catalog: CatalogInterface,
        plans_dir: Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._plans_dir = plans_dir or dc_config.plans_dir
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._plans: dict[str, AcquisitionPlan] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        for f in self._plans_dir.glob("*.json"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                plan = AcquisitionPlan(**data)
                self._plans[plan.plan_id] = plan
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load plan %s: %s", f.name, exc)

    def _save(self, plan: AcquisitionPlan) -> None:
        path = self._plans_dir / f"{plan.plan_id}.json"
        with path.open("w", encoding="utf-8") as f:
            f.write(plan.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def create_plan(self, plan: AcquisitionPlan) -> AcquisitionPlan:
        with self._lock:
            # Check active plan limits
            active_count = sum(
                1
                for p in self._plans.values()
                if p.status in ("draft", "active", "in_progress")
            )
            if active_count >= dc_config.max_active_acquisitions:
                raise AcquisitionLimitError(
                    f"Max active acquisitions reached ({dc_config.max_active_acquisitions})"
                )

            # Validate entries exist in catalog
            for eid in plan.entry_ids:
                if self._catalog.get_entry(eid) is None:
                    raise EntryNotFoundError(
                        f"Cannot create plan: entry not found in catalog: {eid}"
                    )

            # Validate priority
            if plan.priority < dc_config.min_acquisition_priority_score:
                raise AcquisitionError(
                    f"Priority {plan.priority} below min {dc_config.min_acquisition_priority_score}"
                )

            # Validate budget
            total_budget: float = 0.0
            for eid in plan.entry_ids:
                entry = self._catalog.get_entry(eid)
                if entry:
                    total_budget += entry.estimated_budget
            if total_budget > dc_config.acquisition_budget_limit:
                b = dc_config.acquisition_budget_limit
                raise AcquisitionError(
                    f"Total estimated budget {total_budget} exceeds limit {b}"
                )

            self._plans[plan.plan_id] = plan
            self._save(plan)
            return plan

    def get_plan(self, plan_id: str) -> AcquisitionPlan | None:
        return self._plans.get(plan_id)

    def update_plan(self, plan: AcquisitionPlan) -> AcquisitionPlan:
        with self._lock:
            if plan.plan_id not in self._plans:
                raise AcquisitionError(f"Plan not found: {plan.plan_id}")
            updated = AcquisitionPlan(
                plan_id=plan.plan_id,
                entry_ids=plan.entry_ids,
                priority=plan.priority,
                status=plan.status,
                estimated_budget=plan.estimated_budget,
                estimated_storage_mb=plan.estimated_storage_mb,
                target_domains=plan.target_domains,
                notes=plan.notes,
                created_by=plan.created_by,
                created_at=plan.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._plans[plan.plan_id] = updated
            self._save(updated)
            return updated

    def list_active_plans(self) -> list[AcquisitionPlan]:
        return [
            p
            for p in self._plans.values()
            if p.status in ("draft", "active", "in_progress")
        ]

    def prioritize_plan(self, plan_id: str) -> AcquisitionPlan:
        plan = self.get_plan(plan_id)
        if plan is None:
            raise AcquisitionError(f"Plan not found: {plan_id}")

        # Recalculate priority based on entry scores
        if not plan.entry_ids:
            return plan

        scores: list[float] = []
        for eid in plan.entry_ids:
            entry = self._catalog.get_entry(eid)
            if entry:
                scores.append(entry.overall_score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        new_priority = min(avg_score, 1.0)

        updated = AcquisitionPlan(
            plan_id=plan.plan_id,
            entry_ids=plan.entry_ids,
            priority=new_priority,
            status=plan.status,
            estimated_budget=plan.estimated_budget,
            estimated_storage_mb=plan.estimated_storage_mb,
            target_domains=plan.target_domains,
            notes=plan.notes,
            created_by=plan.created_by,
            created_at=plan.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        return self.update_plan(updated)
