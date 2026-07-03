"""Fused-intelligence and threat-assessment models."""

from pydantic import BaseModel, ConfigDict, Field

from backend.contracts.enums.core import ThreatLevel


class FusionResult(BaseModel):
    """Aggregated intelligence picture produced by the Fusion Agent."""

    model_config = ConfigDict(frozen=True)

    combined_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated confidence across all intelligence sources"
    )
    summary: str = Field(
        ..., min_length=1, description="Concise summary of the fused intelligence picture"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence items that support the fused assessment"
    )


class ThreatAssessment(BaseModel):
    """Assessment of threat severity based on fused intelligence."""

    model_config = ConfigDict(frozen=True)

    threat_level: ThreatLevel = Field(..., description="Assessed threat severity")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the threat-level assignment"
    )
    reason: str = Field(
        ..., min_length=1, description="Reasoning behind the threat-level assignment"
    )
