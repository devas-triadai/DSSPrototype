# DSSPrototype

**AI Decision Support System — Prototype Foundation**

---

## Purpose

DSSPrototype is a modular AI Decision Support System designed to ingest multi-source intelligence (imagery, terrain, friendly/enemy forces), fuse it into a coherent battlespace picture, and generate course-of-action recommendations for military or tactical decision-makers.

This repository currently contains **only the project skeleton** — a clean, extensible architecture that future AI modules, services, and a dashboard will plug into without structural changes.

---

## Architecture

The system follows a **layered modular architecture**:

```
┌──────────────────────────────────────────────────┐
│                   Frontend                       │  (future — dashboard UI)
├──────────────────────────────────────────────────┤
│                REST API Layer                    │  backend/api/
├──────────────────────────────────────────────────┤
│               Service Layer                      │  backend/services/
├──────────────────────────────────────────────────┤
│                                                   │
│   ┌────────────┐    ┌───────────────────────┐   │
│   │ Computer   │    │    Knowledge           │   │  backend/modules/
│   │ Vision     │    │   ┌────────┐          │   │
│   │            │    │   │Friendly│          │   │
│   │ (detection)│───▶│   ├────────┤          │   │
│   └────────────┘    │   │ Enemy  │          │   │
│                     │   ├────────┤          │   │
│                     │   │ Terrain│          │   │
│                     │   ├────────┤          │   │
│                     │   │  ...   │          │   │
│                     │   └────────┘          │   │
│                     └────────┬──────────────┘   │
│                              │                   │
│                     ┌────────▼────────┐          │
│                     │  Fusion Engine  │          │
│                     │  (correlation)  │          │
│                     └────────┬────────┘          │
│                              │                   │
│                     ┌────────▼────────┐          │
│                     │ Decision Engine │          │
│                     │ (recommendation)│          │
│                     └────────┬────────┘          │
│                              │                   │
│                     ┌────────▼────────┐          │
│                     │  Orchestration  │          │
│                     │  (pipeline)     │          │
│                     └─────────────────┘          │
├──────────────────────────────────────────────────┤
│           Database / Knowledge Base              │  backend/database/
│                                                  │  knowledge_base/
├──────────────────────────────────────────────────┤
│            Configuration & Logging               │  backend/config/
│                                                  │  backend/core/
└──────────────────────────────────────────────────┘
```

### Design principles

- **Isolation** — Each module is a self-contained Python sub-package with a well-defined interface (`BaseModule`).
- **Extensibility** — New capabilities are added by creating a new sub-package in `backend/modules/` and registering it with the Orchestrator.
- **Testability** — Every module can be unit-tested independently.
- **Configurability** — All settings are driven by environment variables via `pydantic-settings`.
- **No vendor lock-in** — The architecture is framework-agnostic at the module layer.

---

## Folder structure

```
DSSPrototype/
│
├── backend/                        # Python backend package
│   ├── app/                        # FastAPI application entry point
│   │   └── main.py                 # App factory, startup/shutdown hooks
│   ├── api/                        # REST API routers
│   │   ├── __init__.py
│   │   └── health.py               # Health-check endpoint
│   ├── config/                     # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py             # Pydantic Settings class
│   ├── core/                       # Foundational infrastructure
│   │   ├── __init__.py             # Re-exports key helpers
│   │   ├── base.py                 # BaseModule abstract class
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   └── logging.py              # Centralized logger setup
│   ├── services/                   # Future service layer
│   ├── modules/                    # AI capability groups
│   │   ├── computer_vision/        # (future) Object detection / classification
│   │   ├── orchestration/          # (future) Pipeline lifecycle coordinator
│   │   ├── knowledge/              # (future) Domain-specific intelligence agents
│   │   │   ├── friendly/           # Friendly-force analysis
│   │   │   ├── enemy/              # Enemy-force analysis
│   │   │   └── terrain/            # Terrain analysis
│   │   ├── fusion_engine/          # (future) Multi-source intelligence fusion
│   │   └── decision_engine/        # (future) Course-of-action recommendation
│   ├── database/                   # (future) ORM, migrations, connections
│   ├── models/                     # (future) SQLAlchemy / ORM models
│   ├── schemas/                    # (future) Pydantic request/response schemas
│   ├── utils/                      # (future) Shared helpers and decorators
│   ├── logs/                       # Log output directory
│   ├── uploads/                    # File upload directory
│   └── tests/                      # Test suite
│       ├── conftest.py
│       └── test_health.py
│
├── knowledge_base/                 # Structured intelligence storage
│   ├── friendly/                   # (future) Friendly-force KB
│   ├── enemy/                      # (future) Enemy-force KB
│   └── terrain/                    # (future) Terrain KB
│
├── datasets/                       # Raw and processed datasets
│   ├── images/                     # Satellite / drone imagery
│   ├── maps/                       # Map tiles and geospatial data
│   └── documents/                  # PDFs, reports, text
│
├── configs/                        # Static configuration files (YAML, JSON)
├── docs/                           # Project documentation
├── scripts/                        # Utility scripts (dev, deploy, data)
├── assets/                         # Static assets (logos, icons)
├── frontend/                       # (future) Dashboard UI
│
├── .env.example                    # Environment variable template
├── .gitignore
├── pyproject.toml                  # Build config & tool settings
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Future development roadmap

| Phase | Deliverable |
|-------|-------------|
| **0** | Project skeleton *(this phase — complete)* |
| **1** | Configuration system, logging, error handling, base classes |
| **2** | Database schema, migrations, repository layer |
| **3** | Computer Vision service (satellite / drone imagery) |
| **4** | Knowledge intelligence agents (friendly, enemy, terrain) |
| **5** | Fusion Engine (multi-source correlation) |
| **6** | Decision Engine (course-of-action generation) |
| **7** | Orchestration Engine (pipeline management) |
| **8** | REST API endpoints for each module |
| **9** | Knowledge base population & RAG pipelines |
| **10** | Frontend dashboard |
| **11** | Integration testing, hardening, deployment |

---

## How to install

### Prerequisites

- Python **3.11+**
- pip / venv (recommended)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd DSSPrototype

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) Install dev extras
pip install -r requirements.txt  # dev extras included
```

---

## How to start development

```bash
# Run the health-check server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Verify the health endpoint
curl http://localhost:8000/api/v1/health

# Run tests
pytest

# Lint with ruff
ruff check backend/

# Type-check with mypy
mypy backend/
```

The server will respond at `http://localhost:8000/api/v1/health` with:

```json
{
  "status": "healthy",
  "service": "DSSPrototype",
  "version": "0.1.0"
}
```

---

## How future modules plug in

1. **Identify the group** — perception (`computer_vision/`), knowledge (`knowledge/`), or decision (`fusion_engine/` / `decision_engine/`).
2. **Create a new sub-package** under the appropriate group.
3. **Implement** the corresponding interface from `backend.contracts.interfaces`.
4. **Implement** `BaseModule.initialize()` and `BaseModule.shutdown()` from `backend.core.base`.
5. **Define schemas** in `backend/schemas/` for API I/O if needed.
6. **Add tests** in `backend/tests/` mirroring the module path.

For example, adding a weather-analysis domain requires:
1. Create `backend/modules/knowledge/weather/`
2. Create a class implementing `backend.contracts.interfaces.TerrainModule` (or a new interface)
3. No existing code needs to be modified

The architecture is **open for extension, closed for modification**.

---

## Current project status

**Phase 0 — Foundation** *(complete)*

- [x] Project skeleton with complete folder hierarchy
- [x] Modular Python package structure
- [x] Configuration management via `pydantic-settings` / `.env`
- [x] Centralized logging setup
- [x] Reusable error-handling hierarchy
- [x] `BaseModule` abstract class for future AI modules
- [x] Health-check endpoint
- [x] Test suite skeleton with passing health-check test
- [x] `pyproject.toml` with linting / type-checking tool configuration
- [x] `.gitignore`, `.env.example`

**Phase 1 — Contracts & Architecture** *(complete)*

- [x] Strongly typed inter-module communication layer (`backend/contracts/`)
- [x] Enums, data models, abstract interfaces, events, responses, validators
- [x] Architecture refined to enterprise-grade modular structure
- [x] Knowledge agents grouped under `knowledge/` namespace
- [x] Fusion and Decision promoted to standalone engines
- [x] Orchestration as pipeline coordinator
- [x] Package-level README files documenting every module's responsibility

---

## License

Proprietary — all rights reserved.
