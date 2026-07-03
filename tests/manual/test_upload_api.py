"""End-to-end HTTP API test."""
import requests, json

with open("tests/bus.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/runtime/upload",
        files={"file": ("bus.jpg", f, "image/jpeg")},
        timeout=120,
    )

print(f"Status code: {resp.status_code}")
data = resp.json()
print(f"Status: {data['status']}")
print(f"Duration: {data['total_duration_ms']:.1f}ms")
errors = data.get("errors")
if errors:
    print(f"Errors: {errors}")

result = data.get("result", {})
print(f"\nDetection: {len(result.get('detection', {}).get('objects', []))} object(s)")
for obj in result.get("detection", {}).get("objects", []):
    print(f"  [{obj['object_type']}] conf={obj['confidence']:.3f} label={obj['label']}")

friendly = result.get("friendly", {})
print(f"\nFriendly: match={friendly.get('friendly_match')} conf={friendly.get('confidence', 0):.3f}")

enemy = result.get("enemy", {})
print(f"Enemy: match={enemy.get('enemy_match')} conf={enemy.get('confidence', 0):.3f}")

terrain = result.get("terrain", {})
print(f"Terrain: type={terrain.get('terrain_type')}")

fusion = result.get("fusion", {})
print(f"Fusion: conf={fusion.get('combined_confidence', 0):.3f}")

threat = result.get("threat", {})
print(f"Threat: level={threat.get('threat_level')} conf={threat.get('confidence', 0):.3f}")

decision = result.get("decision", {})
print(f"Decision: priority={decision.get('priority')} actions={decision.get('recommended_actions')}")

print(f"\nAll 7 stages present: {all(k in result for k in ['detection','friendly','enemy','terrain','fusion','threat','decision'])}")
