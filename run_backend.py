"""Start the DSSPrototype backend server."""
import sys
sys.path.insert(0, ".")

import uvicorn
from backend.app.main import app

uvicorn.run(app, host="0.0.0.0", port=8000)
