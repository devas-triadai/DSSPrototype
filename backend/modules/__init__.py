"""AI module registry package.

Each top-level sub-package is a self-contained capability group:

- computer_vision /   — image ingestion and object detection
- orchestration /     — pipeline lifecycle and module coordination
- knowledge /         — domain-specific intelligence agents
- fusion_engine /     — multi-source intelligence correlation
- decision_engine /   — course-of-action recommendation

Modules communicate exclusively through types defined in
``backend.contracts`` and implement ``BaseModule`` from
``backend.core.base``.
"""
