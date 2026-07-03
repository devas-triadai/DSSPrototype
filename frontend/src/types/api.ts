import type {
  DecisionCapabilitiesResponse,
  DetectionResult,
  ExecuteResponse,
  FriendlyAnalysis,
  EnemyAnalysis,
  TerrainAnalysis,
  FusionResult,
  ThreatAssessment,
  DecisionRecommendation,
  PipelineStatusResponse,
  SystemInfoResponse,
} from "./contract";

export interface ApiSuccessResponse<T = Record<string, unknown>> {
  success: true;
  message: string;
  data: T | null;
}

export interface ApiErrorResponse {
  success: false;
  error_code: string;
  message: string;
  details: Record<string, unknown> | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface SystemLiveStatus {
  healthy: boolean;
  modules_registered: number;
  modules: string[];
  pipeline_ready: boolean;
  missing_modules?: string[];
  dependency_issues?: string[];
}

export interface VisionValidateResponse {
  is_valid: boolean;
  errors: string[];
  module_available: boolean;
}

export type {
  SystemInfoResponse,
  ExecuteResponse,
  PipelineStatusResponse,
  DecisionCapabilitiesResponse,
};

export type {
  DetectionResult,
  FriendlyAnalysis,
  EnemyAnalysis,
  TerrainAnalysis,
  FusionResult,
  ThreatAssessment,
  DecisionRecommendation,
};
