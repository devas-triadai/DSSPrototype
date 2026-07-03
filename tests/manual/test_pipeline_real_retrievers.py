"""Functional pipeline verification with real retrievers."""
import sys
sys.path.insert(0, ".")

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from backend.contracts.models.detection import DetectionResult, ImageMetadata

# Import all services with real retrievers
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

    print("=== Pipeline Configuration ===")
    print(f"Friendly retriever: {type(friendly_service._retriever).__name__}")
    print(f"Enemy retriever:    {type(enemy_service._retriever).__name__}")
    print(f"Terrain retriever:  {type(terrain_service._retriever).__name__}")

    # Load and process a test image
    test_img = Path("tests/test_image.jpg")
    if not test_img.exists():
        from PIL import Image
        Image.new("RGB", (640, 480), color="red").save(str(test_img))

    img = ImageMetadata(
        image_id=str(test_img),
        timestamp=datetime.now(timezone.utc),
        width=640,
        height=480,
        format="JPEG",
    )

    print("\n=== Executing Pipeline ===")
    result = await runtime.execute(img)

    meta = result.metadata
    print(f"\nPipeline status: {meta.status}")
    print(f"Total duration: {meta.total_duration_ms:.1f}ms")
    print(f"Stage durations: {meta.stage_durations}")
    if meta.errors:
        print(f"Errors: {meta.errors}")

    print("\n=== Stage Results ===")
    print(f"Detection:  {'OK' if result.detection else 'NONE'}")
    if result.detection:
        print(f"  objects: {len(result.detection.objects)}")
        for obj in result.detection.objects:
            print(f"    - {obj.object_type.value} (conf={obj.confidence:.3f})")

    print(f"Friendly:   {'OK' if result.friendly else 'NONE'}")
    if result.friendly:
        print(f"  match={result.friendly.friendly_match} conf={result.friendly.confidence:.3f}")
        print(f"  reason={result.friendly.reason[:120]}...")

    print(f"Enemy:      {'OK' if result.enemy else 'NONE'}")
    if result.enemy:
        print(f"  match={result.enemy.enemy_match} conf={result.enemy.confidence:.3f}")
        if result.enemy.possible_equipment:
            print(f"  equipment={result.enemy.possible_equipment}")
        print(f"  reason={result.enemy.reason[:120]}...")

    print(f"Terrain:    {'OK' if result.terrain else 'NONE'}")
    if result.terrain:
        print(f"  type={result.terrain.terrain_type.value}")
        print(f"  features={result.terrain.nearby_features}")
        print(f"  visibility={result.terrain.visibility}")
        print(f"  road_access={result.terrain.road_access}")
        print(f"  reason={result.terrain.reason[:120]}...")

    print(f"Fusion:     {'OK' if result.fusion else 'NONE'}")
    if result.fusion:
        print(f"  conf={result.fusion.combined_confidence:.3f}")
        print(f"  summary={result.fusion.summary[:120]}...")

    print(f"Threat:     {'OK' if result.threat else 'NONE'}")
    if result.threat:
        print(f"  level={result.threat.threat_level.value} conf={result.threat.confidence:.3f}")

    print(f"Decision:   {'OK' if result.decision else 'NONE'}")
    if result.decision:
        print(f"  priority={result.decision.priority}")
        print(f"  actions={result.decision.recommended_actions}")
        print(f"  reason={result.decision.reason[:120]}...")

    # Verify no NullRetriever
    has_null = any(
        "NullRetriever" in str(type(getattr(service, s, None)))
        for service, s in [
            (friendly_service, "_retriever"),
            (enemy_service, "_retriever"),
            (terrain_service, "_retriever"),
        ]
    )
    print(f"\nNullRetriever detected: {has_null}")

    if has_null:
        print("FAILURE: NullRetriever still in use")
    else:
        print("SUCCESS: All real retrievers are active")

    # Verify all stages produced results
    all_ok = all([
        result.detection is not None,
        result.friendly is not None,
        result.enemy is not None,
        result.terrain is not None,
        result.fusion is not None,
        result.threat is not None,
        result.decision is not None,
    ])
    if all_ok:
        print("SUCCESS: All 7 stages produced valid results")
    else:
        print("FAILURE: Some stages returned None")


asyncio.run(main())
