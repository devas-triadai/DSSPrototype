import apiClient from "./api";
import type {
  PipelineStatusResponse,
  ApiSuccessResponse,
} from "../types/api";

export async function getPipelineStatus(): Promise<PipelineStatusResponse> {
  const { data } = await apiClient.get<
    ApiSuccessResponse<PipelineStatusResponse>
  >("/pipeline/status");
  if (!data.data) {
    throw new Error("Pipeline status not available");
  }
  return data.data;
}
