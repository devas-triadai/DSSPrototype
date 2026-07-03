# Fusion Engine

## Purpose

Merges intelligence from all three knowledge modules (Friendly, Enemy, Terrain) into a single coherent operational picture. Correlates multi-source data, resolves conflicts, and produces combined confidence scoring.

Contains zero AI, zero decision making, zero computer vision, zero knowledge retrieval. All fusion logic is injected via interfaces.

## Architecture

```
                       ┌──────────────────────────────────┐
                       │         FusionService            │
                       │   (public entry point)           │
                       │   implements FusionModule        │
                       └────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
     │    Friendly     │  │      Enemy      │  │    Terrain      │
     │   Analysis      │  │   Analysis      │  │   Analysis      │
     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │          Collector               │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │          Validator                │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │     CorrelationEngine            │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │      ConflictResolver            │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │      SituationBuilder            │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │      ConfidenceScorer            │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │          FusionResult            │
                       │     + ThreatAssessment           │
                       │     (contract models)            │
                       └──────────────────────────────────┘
```

## Execution Pipeline

```
FriendlyAnalysis + EnemyAnalysis + TerrainAnalysis
    │
    ▼
FusionService.fuse_intelligence()
    │
    ├─► Collector.collect(friendly, enemy, terrain)
    │      └─► Packages into CollectedIntelligence
    │
    ├─► Validator.validate(collected)
    │      └─► Checks present, non-empty, in-range
    │
    ├─► CorrelationEngine.correlate(collected)
    │      └─► Correlates friendly↔enemy, enemy↔terrain, friendly↔terrain
    │      └─► Produces CorrelatedEvidence (correlations + supporting evidence)
    │
    ├─► ConflictResolver.resolve(collected, evidence)
    │      └─► Detects identification conflicts, terrain conflicts, confidence conflicts
    │      └─► Produces ConflictRecord (conflicts + resolutions)
    │
    ├─► SituationBuilder.build(collected, evidence, conflicts)
    │      └─► Builds summary, observations, context
    │      └─► Produces SituationReport
    │
    ├─► ConfidenceScorer.score(friendly, enemy, terrain, correlation)
    │      └─► Weighted average → combined_confidence
    │
    └─► FusionResult(combined_confidence, summary, supporting_evidence)
```

### Threat Assessment

```
FusionResult
    │
    ▼
FusionService.assess_threat(fusion)
    │
    ├─► Determine threat level from summary keywords + confidence
    └─► ThreatAssessment(threat_level, confidence, reason)
```

## Inputs

All three from `backend.contracts.models.analysis`:

- `FriendlyAnalysis` — `friendly_match`, `confidence`, `reason`
- `EnemyAnalysis` — `enemy_match`, `confidence`, `possible_equipment`, `reason`
- `TerrainAnalysis` — `terrain_type`, `nearby_features`, `visibility`, `road_access`, `elevation`, `reason`

## Outputs

Both from `backend.contracts.models.fusion`:

- `FusionResult` — `combined_confidence`, `summary`, `supporting_evidence`
- `ThreatAssessment` — `threat_level` (ThreatLevel enum), `confidence`, `reason`

## Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Service** | Public entry point, DI wiring, threat assessment | Correlation, conflict resolution, scoring |
| **Collector** | Package three analyses | Validation, correlation, confidence |
| **Validator** | Check completeness and correctness | Correlation, scoring |
| **CorrelationEngine** | Find cross-domain relationships | Confidence, recommendations |
| **ConflictResolver** | Detect and resolve disagreements | Recommendations, scoring |
| **SituationBuilder** | Build Common Operational Picture | Recommendations, scoring |
| **ConfidenceScorer** | Calculate combined confidence | Correlation, conflict resolution |

## Correlation Rules

| Source A | Source B | Correlation |
|----------|----------|-------------|
| Friendly (match) | Enemy (match) | Possible IFF ambiguity |
| Enemy (match) | Terrain (no road) | Mobility limitation |
| Friendly (match) | Terrain (obscured) | Increased misidentification risk |
| H: Enemy match | — | Elevated threat assessment |
| H: Friendly match | — | Lower threat assessment |

## Conflict Types Detected

| Conflict | Detection | Resolution |
|----------|-----------|------------|
| Identification | Both friendly_match and enemy_match | Higher confidence prevails |
| Terrain | Vehicle in no-road area | Best source trusted |
| Confidence | >0.5 gap between domains | Caution flag added |

## Future Multi-Sensor Fusion

Replace `CorrelationEngineInterface` with a multi-sensor implementation:
```python
class MultiSensorCorrelation(CorrelationEngineInterface):
    def correlate(self, collected) -> CorrelatedEvidence:
        # Fuse radar, EO/IR, SIGINT, and HUMINT-derived assessments
        ...
```

## Future Probabilistic Fusion

Replace `ConfidenceScorerInterface`:
```python
class ProbabilisticScorer(ConfidenceScorerInterface):
    def score(self, friendly, enemy, terrain, correlation) -> float:
        # P(threat | friendly, enemy, terrain) using probability theory
        ...
```

## Future Bayesian Fusion

Replace `ConfidenceScorerInterface`:
```python
class BayesianScorer(ConfidenceScorerInterface):
    def score(self, friendly, enemy, terrain, correlation) -> float:
        prior = 0.5
        likelihood = self._compute_likelihood(friendly, enemy, terrain)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - prior))
        return posterior
```
Config provides `bayesian_prior_friendly`, `bayesian_prior_enemy`, `bayesian_prior_terrain`.

## Future Graph-based Fusion

Replace `CorrelationEngineInterface` and `ConflictResolverInterface`:
```python
class GraphCorrelationEngine(CorrelationEngineInterface):
    def __init__(self, graph_db):
        # Connect to knowledge graph with entity relationships
        ...
    def correlate(self, collected) -> CorrelatedEvidence:
        # Traverse graph edges: supports, contradicts, corroborates
        # Aggregate evidence from entity relationships
        ...
```

## Future Evidence Fusion

Replace `SituationBuilderInterface` for Dempster-Shafer or DST-based fusion:
```python
class EvidenceFusionBuilder(SituationBuilderInterface):
    def build(self, collected, evidence, conflicts) -> SituationReport:
        # Apply Dempster-Shafer theory to combine evidence masses
        # Handle uncertainty and conflict explicitly
        ...
```

## Extension Guide

### How to add a new Correlation Strategy

```python
from backend.modules.fusion_engine.interfaces import (
    CorrelationEngineInterface, CollectedIntelligence, CorrelatedEvidence,
)

class CustomCorrelation(CorrelationEngineInterface):
    def correlate(self, collected) -> CorrelatedEvidence:
        # Custom logic
        ...
```

Then inject:
```python
service = FusionService(correlation_engine=CustomCorrelation())
```

### How to add a new Confidence Scorer

```python
from backend.modules.fusion_engine.interfaces import ConfidenceScorerInterface

class CustomScorer(ConfidenceScorerInterface):
    def score(self, friendly, enemy, terrain, correlation) -> float:
        # Custom algorithm
        ...
```

Then inject:
```python
service = FusionService(confidence_scorer=CustomScorer())
```

## Design Principles

- **Dependency Injection** — all components receive their dependencies via constructors
- **Single Responsibility** — each class does exactly one thing
- **Open/Closed** — add correlation engines and scorers without modifying existing code
- **Strong Typing** — all inter-component data uses dataclasses or contract models
- **Async-First** — all analysis paths are async
- **No Hardcoded Rules** — all fusion logic is injected behind interfaces
- **No AI** — this module provides the framework; probabilistic/Bayesian/graph-based fusion are injected behind interfaces when needed
