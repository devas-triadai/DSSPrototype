"""FastAPI dependency injection for the DSS API layer.

Provides the ``Runtime`` instance to every endpoint.
The runtime is initialised at application startup with all
registered modules.
"""

from fastapi import Request

from backend.modules.orchestration import Runtime

# ---------------------------------------------------------------------------
# Runtime holder — set during application startup
# ---------------------------------------------------------------------------

_runtime: Runtime | None = None


def set_runtime(instance: Runtime) -> None:
    """Store the runtime instance for dependency resolution.

    Called once during application startup.
    """
    global _runtime  # noqa: PLW0603
    _runtime = instance


def get_runtime() -> Runtime:
    """Retrieve the global runtime instance.

    Raises
    ------
    RuntimeError
        If the runtime has not been initialised.
    """
    if _runtime is None:
        raise RuntimeError(
            "Runtime not initialised. "
            "Ensure set_runtime() is called during application startup."
        )
    return _runtime


# ---------------------------------------------------------------------------
# FastAPI-compatible dependency callable
# ---------------------------------------------------------------------------


async def provide_runtime(request: Request) -> Runtime:  # noqa: ARG001
    """FastAPI dependency that yields the global ``Runtime`` instance.

    Usage in route handlers::

        @router.post("/execute")
        async def execute(
            payload: Payload,
            runtime: Runtime = Depends(provide_runtime),
        ):
            return await runtime.execute(payload)
    """
    return get_runtime()
