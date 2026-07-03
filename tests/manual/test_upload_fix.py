"""Verify the UploadFile + File(...) fix for /runtime/upload."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from backend.api.dependencies import set_runtime
from backend.app.main import app
from backend.modules.computer_vision import ComputerVisionService
from backend.modules.decision_engine.service import DecisionService
from backend.modules.fusion_engine.service import FusionService
from backend.modules.knowledge.enemy.service import EnemyKnowledgeService
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService
from backend.modules.knowledge.terrain.service import TerrainKnowledgeService
from backend.modules.orchestration import Runtime

cv = ComputerVisionService()
fr = FriendlyKnowledgeService()
en = EnemyKnowledgeService()
te = TerrainKnowledgeService()
fu = FusionService()
de = DecisionService()

rt = Runtime()
rt.register_module("computer_vision", cv)
rt.register_module("friendly", fr)
rt.register_module("enemy", en)
rt.register_module("terrain", te)
rt.register_module("fusion", fu)
rt.register_module("decision", de)
set_runtime(rt)

test_img = Path("tests/test_image.jpg")
if not test_img.exists():
    from PIL import Image
    Image.new("RGB", (640, 480), color="red").save(str(test_img))

client = TestClient(app)
with open(test_img, "rb") as f:
    resp = client.post("/api/v1/runtime/upload", files={"file": ("test.jpg", f, "image/jpeg")})

print(f"Status: {resp.status_code}")
body = resp.json()
print(f"Success: {body.get('success')}")
d = body.get("data")
if d:
    print(f"Pipeline status: {d.get('status')}")
    print(f"Errors: {d.get('errors')}")
    print(f"Detection: {'OK' if d.get('detection') else 'NONE'}")
    for stage in ["friendly", "enemy", "terrain", "fusion", "threat", "decision"]:
        print(f"{stage:12s}: {'OK' if d.get(stage) else 'NONE'}")
else:
    print(f"Message: {body.get('message')}")
