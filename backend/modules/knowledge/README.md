# Knowledge Intelligence Package

## Purpose

Groups all domain-specific intelligence agents under a single, extensible namespace. Each sub-package analyses one dimension of the battlespace and produces a domain-specific assessment.

## Responsibilities

- Friendly-force identification and tracking (`friendly/`)
- Enemy-force identification, tracking, and equipment recognition (`enemy/`)
- Terrain analysis, visibility, and mobility assessment (`terrain/`)

## Inputs

- `DetectionResult` (contracts) from the Computer Vision module

## Outputs

- `FriendlyAnalysis` (contracts)
- `EnemyAnalysis` (contracts)
- `TerrainAnalysis` (contracts)

## Dependencies

- `backend.contracts.models.analysis`
- `backend.contracts.models.detection`
- `backend.contracts.interfaces.friendly.FriendlyModule`
- `backend.contracts.interfaces.enemy.EnemyModule`
- `backend.contracts.interfaces.terrain.TerrainModule`

## Future Modules

This package is designed for extension. New knowledge domains can be added as sibling sub-packages:

- `weather/` — meteorological impact analysis
- `signals/` — SIGINT / electronic warfare data
- `doctrine/` — enemy doctrine pattern matching
- `satellite/` — orbital asset availability
- `logistics/` — supply-chain and reinforcement analysis

Each new domain simply implements the relevant interface and is added as a directory under `knowledge/`. No structural changes are required.

## Design Philosophy

Separation of concerns — each knowledge domain is an independent analysis pipeline. Domains do not import from each other. They share only the common contracts package.
