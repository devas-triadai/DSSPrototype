"""Integration test: upload image and execute pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

test_img = Path("tests/test_image.jpg")
if not test_img.exists():
    from PIL import Image
    img = Image.new("RGB", (640, 480), color="red")
    img.save(str(test_img))

with open(test_img, "rb") as f:
    response = client.post(
        "/api/v1/runtime/upload",
        files={"file": ("test_image.jpg", f, "image/jpeg")},
    )

print(f"Status: {response.status_code}")
body = response.json()
print(f"Success: {body.get('success')}")
d = body.get("data")
if d:
    print(f"Pipeline status: {d.get('status')}")
    print(f"Errors: {d.get('errors')}")
    print(f"Detection: {'OK' if d.get('detection') else 'NONE'}")
    print(f"Friendly:  {'OK' if d.get('friendly') else 'NONE'}")
    print(f"Enemy:     {'OK' if d.get('enemy') else 'NONE'}")
    print(f"Terrain:   {'OK' if d.get('terrain') else 'NONE'}")
    print(f"Fusion:    {'OK' if d.get('fusion') else 'NONE'}")
    print(f"Threat:    {'OK' if d.get('threat') else 'NONE'}")
    print(f"Decision:  {'OK' if d.get('decision') else 'NONE'}")
    print(f"Stage durations: {d.get('stage_durations')}")
    print(f"Total duration: {d.get('total_duration_ms')}ms")
else:
    print(f"Message: {body.get('message')}")
