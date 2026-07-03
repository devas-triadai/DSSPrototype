"""Full pipeline functional test with real images detectable by YOLO."""
import sys; sys.path.insert(0, ".")
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

from backend.contracts.models.detection import DetectionResult, ImageMetadata
from backend.modules.computer_vision import ComputerVisionService
from backend.modules.decision_engine.service import DecisionService
from backend.modules.fusion_engine.service import FusionService
from backend.modules.knowledge.enemy.service import EnemyKnowledgeService
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService
from backend.modules.knowledge.terrain.service import TerrainKnowledgeService
from backend.modules.orchestration import Runtime


async def test_on_image(name: str, path: str):
    from PIL import Image
    pil = Image.open(path)
    w, h = pil.size
    img = ImageMetadata(
        image_id=path,
        timestamp=datetime.now(timezone.utc),
        width=w,
        height=h,
        format=pil.format or "JPEG",
    )

    runtime = Runtime()
    runtime.register_module("computer_vision", ComputerVisionService())
    runtime.register_module("friendly", FriendlyKnowledgeService())
    runtime.register_module("enemy", EnemyKnowledgeService())
    runtime.register_module("terrain", TerrainKnowledgeService())
    runtime.register_module("fusion", FusionService())
    runtime.register_module("decision", DecisionService())

    print(f"\n{'='*60}")
    print(f"Image: {name} ({w}x{h})")
    print(f"{'='*60}")

    result = await runtime.execute(img)
    meta = result.metadata

    print(f"\nStatus: {meta.status}")
    print(f"Duration: {meta.total_duration_ms:.1f}ms")
    if meta.errors:
        print(f"Errors: {meta.errors}")

    det = result.detection
    if det:
        print(f"\nDetection: {len(det.objects)} object(s)")
        for obj in det.objects:
            print(f"  [{obj.object_type.value}] conf={obj.confidence:.3f} box={obj.bounding_box}")
        valid_objects = [o for o in det.objects if o.confidence > 0.3]
        if not valid_objects:
            print("  (no objects above 0.3 confidence)")
    else:
        print("\nDetection: NONE")

    fri = result.friendly
    if fri:
        print(f"\nFriendly: match={fri.friendly_match} conf={fri.confidence:.3f}")
        print(f"  reason: {fri.reason[:150]}...")
    else:
        print("\nFriendly: NONE")

    ene = result.enemy
    if ene:
        print(f"\nEnemy: match={ene.enemy_match} conf={ene.confidence:.3f}")
        print(f"  equipment: {ene.possible_equipment}")
        print(f"  reason: {ene.reason[:150]}...")
    else:
        print("\nEnemy: NONE")

    ter = result.terrain
    if ter:
        print(f"\nTerrain: type={ter.terrain_type.value}")
        print(f"  features: {ter.nearby_features}")
        print(f"  visibility: {ter.visibility}")
        print(f"  road_access: {ter.road_access}")
        print(f"  reason: {ter.reason[:150]}...")
    else:
        print("\nTerrain: NONE")

    fus = result.fusion
    if fus:
        print(f"\nFusion: conf={fus.combined_confidence:.3f}")
        print(f"  summary: {fus.summary[:180]}...")
    else:
        print("\nFusion: NONE")

    thr = result.threat
    if thr:
        print(f"\nThreat: level={thr.threat_level.value} conf={thr.confidence:.3f}")
    else:
        print("\nThreat: NONE")

    dec = result.decision
    if dec:
        print(f"\nDecision: priority={dec.priority}")
        print(f"  actions: {dec.recommended_actions}")
        print(f"  reason: {dec.reason[:180]}...")
    else:
        print("\nDecision: NONE")

    all_ok = all([
        det is not None, fri is not None, ene is not None,
        ter is not None, fus is not None, thr is not None, dec is not None,
    ])
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'} - {name}")
    return all_ok


async def main():
    images = [
        ("bus.jpg", "tests/bus.jpg"),
        ("zidane.jpg", "tests/zidane.jpg"),
    ]
    results = []
    for name, path in images:
        if Path(path).exists():
            ok = await test_on_image(name, path)
            results.append((name, ok))

    print(f"\n\n=== SUMMARY ===")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

asyncio.run(main())
