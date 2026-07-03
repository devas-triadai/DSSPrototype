# Orchestration Engine

## Purpose

Central coordinator of the AI Decision Support System. Responsible **only** for sequencing module execution and moving strongly typed contract objects through the pipeline. Contains zero business logic, AI, or module-specific code.

## Architecture

```
                    ┌──────────────────────────┐
                    │  OrchestrationService    │
                    │  (public entry point)    │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │      Pipeline            │
                    │  (ordered stage list)    │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │      Workflow            │
                    │  (retry / timeout /      │
                    │   cancellation)          │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │       Router             │
                    │  (module registry)       │
                    └──────────────────────────┘
```

## Execution Pipeline

The default pipeline sequences these stages, each passing a contract model to the next:

1. **Computer Vision** → `ImageMetadata` → `DetectionResult`
2. **Friendly Knowledge** → `DetectionResult` → `FriendlyAnalysis`
3. **Enemy Knowledge** → `DetectionResult` → `EnemyAnalysis`
4. **Terrain Knowledge** → `DetectionResult` → `TerrainAnalysis`
5. **Fusion Engine** → `FriendlyAnalysis` + `EnemyAnalysis` + `TerrainAnalysis` → `FusionResult`
6. **Threat Assessment** → `FusionResult` → `ThreatAssessment`
7. **Decision Engine** → `ThreatAssessment` + `FusionResult` → `DecisionRecommendation`

## Responsibilities

### service.py
- Public entry point (`execute()` / `cancel()`)
- Pipeline lifecycle (context creation, state transitions, error handling)
- Dependency wiring via constructor injection

### pipeline.py
- Ordered stage list with `add_stage`, `insert_stage`, `remove_stage`
- Sequential execution — each stage reads from context, writes to context
- Cancellation check between stages

### workflow.py
- Per-stage retry logic with configurable count and delay
- Per-stage timeout via `asyncio.wait_for`
- Graceful handling of `CancelledError`
- Typed exceptions on failure

### router.py
- Module registry: `register(type_key, instance)` / `get(type_key)`
- Prevents duplicate registration
- Provides clear error messages for unregistered types

### context.py
- Request-scoped `PipelineContext` with UUIDs
- Stage result storage (`set_stage_result` / `get_stage_result`)
- Cancellation flag

### state.py
- `PipelineState` enum with 7 states
- `StateMachine` enforcing valid transitions
- Terminal state detection

### interfaces.py
- 4 abstract interfaces: `PipelineInterface`, `WorkflowInterface`, `RouterInterface`, `StageDefinition`
- All components depend on interfaces, never concrete classes

### config.py
- Environment-driven: `ORCH_PIPELINE_TIMEOUT_SECONDS`, `ORCH_DEFAULT_STAGE_TIMEOUT_SECONDS`, `ORCH_DEFAULT_RETRY_COUNT`, etc.

### exceptions.py
- `PipelineException` (base), `WorkflowException`, `RoutingException`, `TimeoutException`, `CancellationException`, `StateTransitionError`

## Module Coordination

Modules never communicate directly. The orchestration engine:

1. Reads a contract model from the pipeline context
2. Looks up the module via the router
3. Calls the appropriate interface method
4. Stores the result contract back in the context
5. Advances to the next stage

```
Context                    Pipeline               Router
  │                          │                      │
  ├─ initial_input ──────────┤                      │
  │                          ├─ stage1.execute() ───┤── get("computer_vision")
  │                          │                      │── vision.process_image()
  ├─ stage1 result ←─────────┤                      │
  │                          ├─ stage2.execute() ───┤── get("friendly")
  │                          │                      │── friendly.analyze()
  ├─ stage2 result ←─────────┤                      │
  │                          └── ...                │
```

## State Machine

```
PENDING → RECEIVED → RUNNING ──→ COMPLETED
                         │  └──→ FAILED
                         └──→ CANCELLED
                   RUNNING → WAITING → RUNNING
                   WAITING ──→ CANCELLED
```

## Error Handling

| Scenario | Exception | Recovery |
|----------|-----------|----------|
| Stage times out | `TimeoutException` | Retry up to `retry_count` |
| Stage raises | `WorkflowException` | Retry up to `retry_count` |
| Module not registered | `RoutingException` | Immediate pipeline failure |
| Invalid state transition | `StateTransitionError` | Immediate pipeline failure |
| Pipeline exceeds total time | `TimeoutException` | Pipeline marked FAILED |
| Explicit cancellation | `CancellationException` | Pipeline marked CANCELLED |

## Retry Strategy

- Configurable per stage via `StageDefinition.retry_count`
- Exponential fixed delay (`retry_delay_seconds`) between attempts
- Timeout resets on each retry
- Exhausting all retries raises `WorkflowException` or `TimeoutException`

## Future Expansion

Adding a new module to the pipeline requires:

1. **Implement** the corresponding interface in `backend.contracts.interfaces`
2. **Create** the module under `backend/modules/knowledge/` or appropriate group
3. **Register** it with the router: `orchestration.router.register("weather", weather_module)`
4. **Add a stage** to the pipeline: `orchestration.add_stage(my_stage_definition)`

No modification to the orchestration engine itself is required. The pipeline is fully configurable at composition time.

## Design Philosophy

- **Single Responsibility** — orchestrates only; no AI, no business logic
- **Open/Closed** — add stages without modifying pipeline internals
- **Dependency Inversion** — all dependencies are injected interfaces
- **Strong Typing** — only contract models flow between stages
- **Async-First** — all execution paths are async
- **No Global State** — every execution creates a fresh `PipelineContext`
