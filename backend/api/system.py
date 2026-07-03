"""System information and status endpoints.

Provides runtime metadata, registered module information,
and pipeline readiness status.  No business logic.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import provide_runtime
from backend.api.responses import ApiSuccessResponse, SystemInfoResponse
from backend.modules.orchestration import Runtime

router = APIRouter(tags=["system"])


@router.get("/system/info")
async def system_info(
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Return system version, registered modules, and configuration.

    No external services required.
    """
    health = runtime.health()

    return ApiSuccessResponse(
        message="System information retrieved",
        data=SystemInfoResponse(
            version="0.1.0",
            environment="development",
            modules_registered=health.modules_registered,
            modules=health.modules,
            pipeline_ready=health.pipeline_ready,
            config=health.config,
        ).model_dump(),
    )


@router.get("/system/status")
async def system_status(
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Return runtime health status including module readiness."""
    health = runtime.health()

    if health.healthy:
        return ApiSuccessResponse(
            message="Runtime is healthy",
            data=health.model_dump() if hasattr(health, "model_dump") else {
                "healthy": health.healthy,
                "modules_registered": health.modules_registered,
                "modules": health.modules,
                "pipeline_ready": health.pipeline_ready,
            },
        )

    return ApiSuccessResponse(
        message="Runtime is degraded",
        data={
            "healthy": health.healthy,
            "modules_registered": health.modules_registered,
            "modules": health.modules,
            "missing_modules": health.missing_modules,
            "dependency_issues": health.dependency_issues,
            "pipeline_ready": health.pipeline_ready,
        },
    )
