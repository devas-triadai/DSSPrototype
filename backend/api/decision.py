"""Decision capabilities endpoint.

Returns information about the Decision Engine's supported
capabilities.  No business logic, no AI.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import provide_runtime
from backend.api.responses import (
    ApiSuccessResponse,
    DecisionCapabilitiesResponse,
)
from backend.modules.orchestration import Runtime

router = APIRouter(tags=["decision"])


@router.get("/decision/capabilities")
async def decision_capabilities(
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Return the Decision Engine's supported capabilities.

    Lists available COA generation strategies, priority range,
    and template sources.  Does not generate recommendations.
    """
    decision_available = False
    try:
        _ = runtime.registry.get_decision()
        decision_available = True
    except RuntimeError:
        pass

    capabilities = [
        "Course-of-Action generation (configurable templates)",
        "Priority assignment (1 highest to 5 lowest)",
        "Situation evaluation (threat + fusion)",
        "Confidence scoring (weighted average)",
        "Explainable recommendation reasoning",
    ]

    if not decision_available:
        capabilities = []

    response = DecisionCapabilitiesResponse(
        available=decision_available,
        capabilities=capabilities,
    )

    return ApiSuccessResponse(
        message="Decision capabilities retrieved",
        data=response.model_dump(),
    )
