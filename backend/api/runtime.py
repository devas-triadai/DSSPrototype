"""Pipeline execution endpoint.

Accepts image metadata, triggers the full DSS pipeline via
the Runtime, and returns the complete execution result.
No business logic.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from backend.api.dependencies import provide_runtime
from backend.api.responses import ApiSuccessResponse, ExecuteResponse
from backend.config.settings import settings
from backend.contracts.models.detection import ImageMetadata
from backend.modules.computer_vision.config import cv_config
from backend.modules.orchestration import Runtime, RuntimeResult

router = APIRouter(tags=["runtime"])

_SUPPORTED_EXTENSIONS: set[str] = {f".{fmt}" for fmt in cv_config.supported_formats}


@router.post("/runtime/execute")
async def execute_pipeline(
    image: ImageMetadata,
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Execute the full DSS pipeline on the provided image metadata.

    Accepts an ``ImageMetadata`` payload, runs all seven pipeline
    stages (Computer Vision → Knowledge → Fusion → Decision),
    and returns the results with execution timing.
    """
    result: RuntimeResult = await runtime.execute(image)

    response = _build_execute_response(result)

    if result.success:
        return ApiSuccessResponse(
            message="Pipeline execution completed",
            data=response.model_dump(),
        )

    return ApiSuccessResponse(
        message="Pipeline execution failed",
        data=response.model_dump(),
    )


@router.post("/runtime/upload")
async def upload_and_execute(
    file: UploadFile = File(...),
    runtime: Runtime = Depends(provide_runtime),
) -> ApiSuccessResponse:
    """Upload an image and execute the full DSS pipeline.

    Accepts a ``multipart/form-data`` image file, saves it to the
    uploads folder, auto-generates ``ImageMetadata`` (UUID, timestamp,
    dimensions, format), triggers all seven pipeline stages, and
    returns the execution result.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = _get_extension(file.filename)
    if ext.lower() not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    image_id = str(uuid.uuid4())
    upload_path = settings.upload_folder / f"{image_id}{ext}"
    settings.upload_folder.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    upload_path.write_bytes(contents)

    try:
        with Image.open(upload_path) as img:
            width, height = img.size
            img_format = img.format or ext.lstrip(".").upper()
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    metadata = ImageMetadata(
        image_id=str(upload_path),
        timestamp=datetime.now(timezone.utc),
        source="local_upload",
        width=width,
        height=height,
        format=img_format,
    )

    try:
        result: RuntimeResult = await runtime.execute(metadata)
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise

    response = _build_execute_response(result)

    if result.success:
        return ApiSuccessResponse(
            message="Pipeline execution completed",
            data=response.model_dump(),
        )

    return ApiSuccessResponse(
        message="Pipeline execution failed",
        data=response.model_dump(),
    )


def _get_extension(filename: str) -> str:
    """Extract the file extension including the dot."""
    idx = filename.rfind(".")
    if idx == -1 or idx == len(filename) - 1:
        return ""
    return filename[idx:]


def _build_execute_response(result: RuntimeResult) -> ExecuteResponse:
    """Convert a ``RuntimeResult`` into an ``ExecuteResponse``."""
    return ExecuteResponse(
        request_id=result.metadata.request_id,
        pipeline_id=result.metadata.pipeline_id,
        status=result.metadata.status,
        total_duration_ms=result.metadata.total_duration_ms,
        stage_durations=result.metadata.stage_durations,
        errors=result.metadata.errors,
        warnings=result.metadata.warnings,
        detection=result.detection,
        friendly=result.friendly,
        enemy=result.enemy,
        terrain=result.terrain,
        fusion=result.fusion,
        threat=result.threat,
        decision=result.decision,
    )
