# Decision Engine

## Purpose

Generates explainable Course-of-Action (COA) recommendations for human commanders based on fused intelligence and threat assessment. This module never issues autonomous orders — its output is always a recommendation for human review.

Contains zero AI, zero autonomous command logic. All decision logic is injected via interfaces.

## Architecture

```
                       ┌──────────────────────────────────┐
                       │        DecisionService           │
                       │   (public entry point)           │
                       │   implements DecisionModule      │
                       └────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐  ┌────────▼────────┐            │
     │   FusionResult  │  │ ThreatAssessment │            │
     └────────┬────────┘  └────────┬────────┘            │
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │      SituationEvaluator          │
                       │   (evaluate operational context) │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │        COAGenerator              │
                       │   (generate courses of action)   │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │       PriorityAnalyzer           │
                       │   (assign 1-5 priority)         │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │      ConfidenceScorer            │
                       │   (calculate recommendation conf)│
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │    RecommendationBuilder         │
                       │   (build DecisionRecommendation) │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │   DecisionRecommendation         │
                       │     (contract model)             │
                       └──────────────────────────────────┘
```

## Execution Pipeline

```
FusionResult + ThreatAssessment
    │
    ▼
DecisionService.generate_recommendations()
    │
    ├─► 1. SituationEvaluator.evaluate(fusion, threat)
    │      └─► Determines: has_enemy, has_friendly, terrain_summary,
    │                       severity, key_observations
    │      └─► Produces: SituationContext
    │
    ├─► 2. COAGenerator.generate(situation, threat)
    │      └─► Loads COA templates from config per threat level
    │      └─► Filters by situation relevance
    │      └─► Produces: list[str] actions
    │
    ├─► 3. PriorityAnalyzer.analyze(situation, threat)
    │      └─► Base priority from threat level
    │      └─► Adjusted by confidence and severity
    │      └─► Produces: int priority (1-5)
    │
    ├─► 4. ConfidenceScorer.score(fusion, threat, situation)
    │      └─► Weighted average: fusion×0.40 + threat×0.40 + situation×0.20
    │      └─► Produces: float confidence
    │
    ├─► 5. RecommendationBuilder.build(situation, actions, priority, confidence)
    │      └─► Synthesises reason string from all inputs
    │      └─► Produces: DecisionRecommendation
    │
    └─► DecisionRecommendation(recommended_actions, priority, reason)
```

### Commander Decision Recording

```
DecisionService.process_decision(commander_decision)
    └─► Logs the commander's chosen COA, operator identity, and remarks
    └─► Future: forward to doctrine engine, rules engine, or human-approval
```

## Inputs

- `FusionResult` — `combined_confidence`, `summary`, `supporting_evidence`
- `ThreatAssessment` — `threat_level` (ThreatLevel), `confidence`, `reason`

## Outputs

- `DecisionRecommendation` — `recommendation_id`, `recommended_actions`, `priority` (1-5), `reason`
- `CommanderDecision` (received, not produced) — `decision`, `decision_time`, `operator_name`, `remarks`

## Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Service** | Public entry point, DI wiring, confidence computation | Situation eval, COA generation, priority |
| **SituationEvaluator** | Assess operational context (enemy/friendly presence, severity) | Recommendations, COA generation |
| **COAGenerator** | Generate relevant courses of action from configurable templates | Doctrine, autonomous decisions |
| **PriorityAnalyzer** | Assign priority 1-5 based on threat + situation | Recommendations |
| **RecommendationBuilder** | Synthesise final recommendation with reasoning | Priority, confidence |
| **ConfidenceScorer** | Calculate combined recommendation confidence | COA generation, priority |

## COA Templates (Configurable)

| Threat Level | Default COAs |
|-------------|--------------|
| CRITICAL | Alert Headquarters, Request Immediate Reinforcement, Prepare Defensive Positions, Activate Contingency Plan |
| HIGH | Alert Headquarters, Request Additional Reconnaissance, Track Target, Monitor Situation |
| MEDIUM | Report to Headquarters, Monitor Situation, Request Additional Reconnaissance |
| LOW | Continue Surveillance, Log Observation |
| UNKNOWN | Continue Surveillance, Request Human Review |

All templates are overridable via `DECISION_COA_TEMPLATES_*` environment variables.

## Priority Mapping

| Threat Level | Base Priority |
|-------------|---------------|
| CRITICAL | 1 |
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 4 |
| UNKNOWN | 5 |

Adjusted by confidence (±0-1) and severity (±0-1), clamped to 1-5.

## Future Doctrine Engine Integration

Replace `COAGeneratorInterface` with a doctrine-aware implementation:

```python
class DoctrineCOAGenerator(COAGeneratorInterface):
    def __init__(self, doctrine_endpoint: str, doctrine_version: str):
        # Connect to military doctrine service
        ...
    def generate(self, situation, threat) -> list[str]:
        # Query doctrine engine for approved COAs based on:
        #   - Current rules of engagement
        #   - Tactical doctrine version
        #   - Operational context
        ...
```

Config already provides `doctrine_endpoint` and `doctrine_version`.

## Future Rules Engine Integration

Replace `COAGeneratorInterface` or `PriorityAnalyzerInterface`:

```python
class RulesEngineCOAGenerator(COAGeneratorInterface):
    def __init__(self, rules_engine_endpoint: str):
        # Connect to business-rules / decision-table engine
        ...
    def generate(self, situation, threat) -> list[str]:
        # Evaluate rules: IF threat=HIGH AND enemy=present THEN COA=alert
        ...
```

Config already provides `rules_engine_type` and `rules_engine_endpoint`.

## Future Human Approval Workflow

Replace `RecommendationBuilderInterface` or extend `DecisionService`:

```python
class HumanApprovalBuilder(RecommendationBuilderInterface):
    def __init__(self, approval_timeout: int, escalation_contact: str):
        # Integration with notification / alerting system
        ...
    def build(self, situation, actions, priority, confidence):
        # 1. Build recommendation
        # 2. Send to human operator for approval
        # 3. Wait for response or escalate
        # 4. Return approved or modified recommendation
        ...
```

Config already provides `require_human_approval`, `approval_timeout_seconds`, and `escalation_contact`.

## Extension Guide

### How to add a new COA Strategy

```python
from backend.modules.decision_engine.interfaces import (
    COAGeneratorInterface, SituationContext, ThreatAssessment,
)

class CustomCOAGenerator(COAGeneratorInterface):
    def generate(self, situation, threat) -> list[str]:
        # Custom COA logic
        ...
```

Then inject:
```python
service = DecisionService(coa_generator=CustomCOAGenerator())
```

### How to add a new Priority Strategy

```python
from backend.modules.decision_engine.interfaces import (
    PriorityAnalyzerInterface, SituationContext, ThreatAssessment,
)

class CustomPriorityAnalyzer(PriorityAnalyzerInterface):
    def analyze(self, situation, threat) -> int:
        # Custom priority logic (e.g. matrix-based)
        ...
```

Then inject:
```python
service = DecisionService(priority_analyzer=CustomPriorityAnalyzer())
```

## Design Principles

- **Dependency Injection** — all components receive their dependencies via constructors
- **Single Responsibility** — each class does exactly one thing
- **Open/Closed** — add COA generators and priority strategies without modifying existing code
- **Strong Typing** — all inter-component data uses dataclasses or contract models
- **Async-First** — all analysis paths are async
- **No Autonomous Decisions** — this module only recommends; command authority always rests with humans
- **Configurable Doctrine** — all COA templates are environment-configurable; no hardcoded tactical rules
