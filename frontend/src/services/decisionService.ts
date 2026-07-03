import apiClient from "./api";
import type {
  DecisionCapabilitiesResponse,
  ApiSuccessResponse,
} from "../types/api";

export async function getDecisionCapabilities(): Promise<DecisionCapabilitiesResponse> {
  const { data } = await apiClient.get<
    ApiSuccessResponse<DecisionCapabilitiesResponse>
  >("/decision/capabilities");
  if (!data.data) {
    throw new Error("Decision capabilities not available");
  }
  return data.data;
}
