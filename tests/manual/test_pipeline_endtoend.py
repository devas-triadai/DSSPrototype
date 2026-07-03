"""End-to-end pipeline integration test.

Exercises the full Runtime pipeline: registration, execution,
and result assembly for all 7 stages.
"""

import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from backend.contracts.models.detection import ImageMetadata
from backend.modules.computer_vision import ComputerVisionService
from backend.modules.decision_engine.service import DecisionService
from backend.modules.fusion_engine.service import FusionService
from backend.modules.knowledge.enemy.service import EnemyKnowledgeService
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService
from backend.modules.knowledge.terrain.service import TerrainKnowledgeService
from backend.modules.orchestration import Runtime


async def main():
    cv_service = ComputerVisionService()
    friendly_service = FriendlyKnowledgeService()
    enemy_service = EnemyKnowledgeService()
    terrain_service = TerrainKnowledgeService()
    fusion_service = FusionService()
    decision_service = DecisionService()

    runtime = Runtime()
    runtime.register_module("computer_vision", cv_service)
    runtime.register_module("friendly", friendly_service)
    runtime.register_module("enemy", enemy_service)
    runtime.register_module("terrain", terrain_service)
    runtime.register_module("fusion", fusion_service)
    runtime.register_module("decision", decision_service)

    count = len(runtime.registry.registered_types())
    print(f"All modules registered ({count}/6): {runtime.registry.is_complete()}")

    img = ImageMetadata(
        image_id="tests/test_image.jpg",
        timestamp=datetime.now(timezone.utc),
        width=640,
        height=480,
        format="jpg",
    )

    result = await runtime.execute(img)

    meta = result.metadata
    print(f"Pipeline status: {meta.status}")
    print(f"Total duration: {meta.total_duration_ms:.1f}ms")
    if meta.errors:
        print(f"  Errors: {meta.errors}")
    if meta.stage_durations:
        print(f"  Stage durations: {meta.stage_durations}")

    fields = ["detection", "friendly", "enemy", "terrain", "fusion", "threat", "decision"]
    all_ok = True
    for field in fields:
        val = getattr(result, field)
        status = "OK" if val is not None else "NONE"
        if status == "NONE":
            all_ok = False
        extra = ""
        if val is not None and hasattr(val, "confidence"):
            extra = f" (conf={val.confidence:.3f})"
        print(f"  {field:12s}: {status}{extra}")

    if all_ok:
        print("\nSUCCESS: All 7 pipeline stages produced valid results")
    else:
        print("\nPARTIAL: Some stages returned None")


asyncio.run(main())
