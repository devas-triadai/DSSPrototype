"""Tests for AcquisitionPlanner."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.acquisition_planner import AcquisitionPlanner
from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.exceptions import (
    AcquisitionError,
    AcquisitionLimitError,
    EntryNotFoundError,
)
from backend.dataset_catalog.models import AcquisitionPlan, CatalogEntry


def _setup_catalog(tmp: Path) -> tuple[Catalog, AcquisitionPlanner]:
    cat = Catalog(tmp / "catalog.json")
    planner = AcquisitionPlanner(cat, plans_dir=tmp / "plans")
    entry = CatalogEntry(
        entry_id="e_001",
        name="Test Dataset",
        source_id="src_001",
        source_type="local",
        overall_score=0.85,
        estimated_budget=500.0,
        estimated_storage_mb=100.0,
    )
    cat.add_entry(entry)
    return cat, planner


def test_create_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        plan = AcquisitionPlan(
            plan_id="plan_001",
            entry_ids=["e_001"],
            priority=0.8,
        )
        result = planner.create_plan(plan)
        assert result.plan_id == "plan_001"
        assert result.status == "draft"


def test_get_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        planner.create_plan(
            AcquisitionPlan(plan_id="plan_001", entry_ids=["e_001"], priority=0.8)
        )
        assert planner.get_plan("plan_001") is not None
        assert planner.get_plan("ghost") is None


def test_update_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        planner.create_plan(
            AcquisitionPlan(plan_id="plan_001", entry_ids=["e_001"], priority=0.8)
        )
        updated = AcquisitionPlan(
            plan_id="plan_001",
            entry_ids=["e_001"],
            priority=0.9,
            status="active",
        )
        result = planner.update_plan(updated)
        assert result.priority == 0.9
        assert result.status == "active"


def test_update_nonexistent_plan_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        try:
            planner.update_plan(
                AcquisitionPlan(plan_id="ghost", entry_ids=[], priority=0.5)
            )
            assert False, "Expected AcquisitionError"
        except AcquisitionError:
            pass


def test_list_active_plans() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        planner.create_plan(
            AcquisitionPlan(plan_id="p1", entry_ids=["e_001"], priority=0.8, status="draft")
        )
        planner.create_plan(
            AcquisitionPlan(plan_id="p2", entry_ids=["e_001"], priority=0.8, status="active")
        )
        planner.create_plan(
            AcquisitionPlan(plan_id="p3", entry_ids=["e_001"], priority=0.8, status="completed")
        )
        active = planner.list_active_plans()
        assert len(active) == 2


def test_prioritize_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, planner = _setup_catalog(tmp_path)
        planner.create_plan(
            AcquisitionPlan(plan_id="plan_001", entry_ids=["e_001"], priority=0.5)
        )
        result = planner.prioritize_plan("plan_001")
        assert result.priority == 0.85  # matches entry overall_score


def test_create_plan_exceeds_active_limit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cat = Catalog(tmp_path / "catalog.json")
        planner = AcquisitionPlanner(cat, plans_dir=tmp_path / "plans")
        entry = CatalogEntry(
            entry_id="e_001",
            name="Test",
            source_id="src_001",
            source_type="local",
        )
        cat.add_entry(entry)

        # Create max active plans
        for i in range(5):
            planner.create_plan(
                AcquisitionPlan(
                    plan_id=f"p_{i}", entry_ids=["e_001"], priority=0.5, status="active"
                )
            )
        # Next one should fail
        try:
            planner.create_plan(
                AcquisitionPlan(
                    plan_id="p_overflow", entry_ids=["e_001"], priority=0.5
                )
            )
            assert False, "Expected AcquisitionLimitError"
        except AcquisitionLimitError:
            pass


def test_create_plan_nonexistent_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cat = Catalog(tmp_path / "catalog.json")
        planner = AcquisitionPlanner(cat, plans_dir=tmp_path / "plans")
        try:
            planner.create_plan(
                AcquisitionPlan(
                    plan_id="p_bad", entry_ids=["ghost"], priority=0.5
                )
            )
            assert False, "Expected EntryNotFoundError"
        except EntryNotFoundError:
            pass


def test_create_plan_below_min_priority() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, planner = _setup_catalog(Path(tmp))
        try:
            planner.create_plan(
                AcquisitionPlan(
                    plan_id="p_low", entry_ids=["e_001"], priority=0.1
                )
            )
            assert False, "Expected AcquisitionError"
        except AcquisitionError:
            pass
