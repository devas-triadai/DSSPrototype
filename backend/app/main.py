"""FastAPI application entry point.

Initializes the web server, registers routers, and
hooks startup / shutdown lifecycle events.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from backend.api.decision import router as decision_router
from backend.api.dependencies import set_runtime
from backend.api.exceptions import (
    general_exception_handler,
    runtime_error_handler,
    validation_exception_handler,
)
from backend.api.health import router as health_router
from backend.api.middleware import RequestLoggingMiddleware
from backend.api.pipeline import router as pipeline_router
from backend.api.runtime import router as execute_router
from backend.api.system import router as system_router
from backend.api.vision import router as vision_router
from backend.config.settings import settings
from backend.core.logging import setup_logging
from backend.modules.computer_vision import ComputerVisionService
from backend.modules.decision_engine.service import DecisionService
from backend.modules.fusion_engine.service import FusionService
from backend.modules.knowledge.enemy.service import EnemyKnowledgeService
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService
from backend.modules.knowledge.terrain.service import TerrainKnowledgeService
from backend.modules.orchestration import Runtime

logger = setup_logging(settings.log_level, settings.log_file)

_API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown."""
    logger.info(
        "%s v%s starting in %s mode …",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )

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
    set_runtime(runtime)
    logger.info(
        "Runtime initialised (%d modules registered)",
        len(runtime.registry.registered_types()),
    )

    yield

    logger.info("%s shutting down …", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Decision Support System Prototype",
    lifespan=lifespan,
)

# Middleware (order matters: outermost first)
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RuntimeError, runtime_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, general_exception_handler)

# Routers
app.include_router(health_router, prefix=_API_V1_PREFIX)
app.include_router(system_router, prefix=_API_V1_PREFIX)
app.include_router(execute_router, prefix=_API_V1_PREFIX)
app.include_router(vision_router, prefix=_API_V1_PREFIX)
app.include_router(pipeline_router, prefix=_API_V1_PREFIX)
app.include_router(decision_router, prefix=_API_V1_PREFIX)
