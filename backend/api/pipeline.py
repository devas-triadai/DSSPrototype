"""Pipeline status endpoint.

Returns the execution pipeline configuration, registered stages,
and execution order.  No business logic.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import provide_runtime
from backend.api.responses import ApiSuccessResponse, PipelineStatusResponse
from backend.modules.orchestration import PipelineExecutor, Runtime

router = APIRouter(tags=["pipeline"])

_STAGE_NAMES: list[str] = [
    "Computer Vision",
    "Friendly Knowledge",
    "Enemy Knowledge",
    "Terrain Knowledge",
    "Fusion Engine",
    "Decision Engine",
]

_EXECUTION_ORDER: list[str] = [
    "computer_vision",
    "friendly",
    "enemy",
    "terrain",
    "fusion",
    "decision",
]


@router.get("/pipeline/status")
async def pipeline_status(
    runtime: Runtime = Depends(provide_runtime),  # noqa: ARG001
) -> ApiSuccessResponse:
    """Return the current pipeline configuration and execution order.

    Provides the list of registered stages, their execution
    order, and groups eligible for future parallel execution.
    """
    parallel_groups = PipelineExecutor.parallel_stage_groups()

    status = PipelineStatusResponse(
        status="configured",
        stage_count=len(_EXECUTION_ORDER),
        execution_order=_EXECUTION_ORDER,
        parallel_groups=parallel_groups,
    )

    return ApiSuccessResponse(
        message="Pipeline status retrieved",
        data=status.model_dump(),
    )
