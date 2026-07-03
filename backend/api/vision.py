"""Vision validation endpoint.

Validates image metadata without running inference.
No business logic, no AI.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import provide_runtime
from backend.api.responses import ApiSuccessResponse
from backend.contracts.models.detection import ImageMetadata
from backend.modules.orchestration import Runtime

router = APIRouter(tags=["vision"])


@router.post("/vision/validate")
async def validate_image(
    image: ImageMetadata,
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Validate image metadata without running inference.

    Checks that the provided ``ImageMetadata`` has the required
    fields (``image_id``, ``timestamp``) and reasonable dimensions.
    Does **not** execute the detection pipeline.
    """
    issues: list[str] = []

    if not image.image_id:
        issues.append("image_id is required")

    if image.width is not None and image.width <= 0:
        issues.append("width must be positive")
    if image.height is not None and image.height <= 0:
        issues.append("height must be positive")

    if image.width is not None and image.height is not None:
        aspect = max(image.width, image.height) / max(1, min(image.width, image.height))
        if aspect > 10.0:
            issues.append("Aspect ratio exceeds 10:1")

    module_available = False
    try:
        _ = runtime.registry.get_vision()
        module_available = True
    except RuntimeError:
        issues.append("Computer Vision module not registered")

    if issues:
        return ApiSuccessResponse(
            message="Image validation failed",
            data={
                "is_valid": False,
                "errors": issues,
                "module_available": module_available,
            },
        )

    return ApiSuccessResponse(
        message="Image metadata is valid",
        data={
            "is_valid": True,
            "errors": [],
            "module_available": module_available,
        },
    )
