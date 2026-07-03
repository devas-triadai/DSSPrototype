"""Domain-event payloads that flow through the DSS pipeline.

Each event records *what happened* and *when*.  These models
are consumed by future pub/sub infrastructure but are defined
here so all modules agree on the schema.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from backend.contracts.models.decision import CommanderDecision, DecisionRecommendation
from backend.contracts.models.detection import DetectionResult
from backend.contracts.models.fusion import ThreatAssessment


class ImageUploaded(BaseModel):
    """Emitted when a new image is ingested into the system."""

    image_id: str = Field(..., description="Unique identifier of the uploaded image")
    timestamp: datetime = Field(..., description="When the upload occurred")
    file_path: str = Field(..., min_length=1, description="Filesystem path to the uploaded image")


class DetectionCompleted(BaseModel):
    """Emitted when the Computer Vision module finishes processing an image."""

    image_id: str = Field(..., description="Foreign key referencing the source image")
    detection_result: DetectionResult = Field(..., description="The completed detection output")
    timestamp: datetime = Field(..., description="When detection finished")


class ThreatIdentified(BaseModel):
    """Emitted when the Fusion Agent identifies a threat."""

    threat_assessment: ThreatAssessment = Field(..., description="The identified threat assessment")
    source_fusion_id: str = Field(
        ..., min_length=1, description="Foreign key to the originating fusion result"
    )
    timestamp: datetime = Field(..., description="When the threat was identified")


class DecisionGenerated(BaseModel):
    """Emitted when the Decision Support Agent produces a recommendation."""

    recommendation: DecisionRecommendation = Field(..., description="The generated recommendation")
    timestamp: datetime = Field(..., description="When the recommendation was generated")


class CommanderApproved(BaseModel):
    """Emitted when a commander approves or rejects a recommendation."""

    decision: CommanderDecision = Field(..., description="The commander's decision")
    recommendation_id: str = Field(
        ..., min_length=1, description="Foreign key to the originating recommendation"
    )
    timestamp: datetime = Field(..., description="When the approval was recorded")
