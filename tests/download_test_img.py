"""Download a test image for YOLO detection testing."""
import sys; sys.path.insert(0, ".")

import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

# Try a Wikimedia image (public domain)
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/9/9f/M1A1_Abrams_Tank.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Leopard_2_A7_%2826688686761%29.jpg/800px-Leopard_2_A7_%2826688686761%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/CH-47_Chinook_helicopter.jpg/800px-CH-47_Chinook_helicopter.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Hummer_H1_%284%29.jpg/800px-Hummer_H1_%284%29.jpg",
]

headers = {"User-Agent": "Mozilla/5.0 (compatible; DSSPrototype/1.0)"}

for url in urls:
    try:
        print(f"Trying: {url}")
        resp = requests.get(url, timeout=15, headers=headers)
        ct = resp.headers.get("Content-Type", "")
        print(f"  Status: {resp.status_code}, Content-Type: {ct}, Size: {len(resp.content)}")
        if resp.status_code == 200 and "image" in ct:
            img = Image.open(BytesIO(resp.content))
            out = Path("tests") / f"test_{Path(url).stem}.jpg"
            img.save(str(out))
            print(f"  Saved: {out} ({img.size})")
        else:
            print(f"  Not an image response")
    except Exception as e:
        print(f"  Error: {e}")
