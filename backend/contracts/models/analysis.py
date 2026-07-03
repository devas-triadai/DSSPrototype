"""Intelligence-analysis models produced by domain agents."""

from pydantic import BaseModel, ConfigDict, Field

from backend.contracts.enums.core import TerrainType


class FriendlyAnalysis(BaseModel):
    """Assessment of whether a detected subject matches known friendly forces."""

    model_config = ConfigDict(frozen=True)

    friendly_match: bool = Field(
        ..., description="Whether the subject matches a known friendly force"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the friendly-match determination"
    )
    reason: str = Field(..., min_length=1, description="Reasoning behind the assessment")


class EnemyAnalysis(BaseModel):
    """Assessment of whether a detected subject matches known enemy forces."""

    model_config = ConfigDict(frozen=True)

    enemy_match: bool = Field(..., description="Whether the subject matches a known enemy force")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the enemy-match determination"
    )
    possible_equipment: str | None = Field(None, description="Identified or suspected equipment")
    reason: str = Field(..., min_length=1, description="Reasoning behind the assessment")


class TerrainAnalysis(BaseModel):
    """Assessment of terrain characteristics for a given area."""

    model_config = ConfigDict(frozen=True)

    terrain_type: TerrainType = Field(..., description="Dominant terrain classification")
    nearby_features: list[str] = Field(
        default_factory=list, description="Notable nearby terrain features"
    )
    visibility: str = Field(
        ..., min_length=1, description="Visibility assessment, e.g. 'good' or 'obscured'"
    )
    road_access: bool = Field(..., description="Whether road access is available")
    elevation: float | None = Field(None, description="Elevation in metres above sea level")
    reason: str = Field(..., min_length=1, description="Reasoning behind the assessment")
