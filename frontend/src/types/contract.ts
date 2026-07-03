import type { ObjectType, TerrainType, ThreatLevel } from "./enums";

export interface ImageMetadata {
  image_id: string;
  timestamp: string;
  source?: string;
  width?: number;
  height?: number;
  format?: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  id: string;
  object_type: ObjectType;
  confidence: number;
  bounding_box: BoundingBox;
  label: string | null;
  description: string | null;
}

export interface DetectionResult {
  image_id: string;
  timestamp: string;
  objects: DetectedObject[];
  model_version: string;
  processing_time_ms: number;
}

export interface FriendlyAnalysis {
  friendly_match: boolean;
  confidence: number;
  reason: string;
}

export interface EnemyAnalysis {
  enemy_match: boolean;
  confidence: number;
  possible_equipment: string | null;
  reason: string;
}

export interface TerrainAnalysis {
  terrain_type: TerrainType;
  nearby_features: string[];
  visibility: string;
  road_access: boolean;
  elevation: number | null;
  reason: string;
}

export interface FusionResult {
  combined_confidence: number;
  summary: string;
  supporting_evidence: string[];
}

export interface ThreatAssessment {
  threat_level: ThreatLevel;
  confidence: number;
  reason: string;
}

export interface DecisionRecommendation {
  recommendation_id: string;
  recommended_actions: string[];
  priority: number;
  reason: string;
}

export interface ExecuteResponse {
  request_id: string;
  pipeline_id: string;
  status: "completed" | "failed";
  total_duration_ms: number;
  stage_durations: Record<string, number>;
  errors: string[];
  warnings: string[];
  detection: DetectionResult | null;
  friendly: FriendlyAnalysis | null;
  enemy: EnemyAnalysis | null;
  terrain: TerrainAnalysis | null;
  fusion: FusionResult | null;
  threat: ThreatAssessment | null;
  decision: DecisionRecommendation | null;
}

export interface SystemInfoResponse {
  version: string;
  environment: string;
  modules_registered: number;
  modules: string[];
  pipeline_ready: boolean;
  config: Record<string, unknown>;
}

export interface PipelineStatusResponse {
  status: string;
  stage_count: number;
  execution_order: string[];
  parallel_groups: string[][];
}

export interface DecisionCapabilitiesResponse {
  available: boolean;
  capabilities: string[];
  priority_range: string;
  coa_sources: string;
}
