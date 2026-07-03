import apiClient from "./api";
import type {
  SystemInfoResponse,
  SystemLiveStatus,
  ApiSuccessResponse,
} from "../types/api";

export async function getSystemInfo(): Promise<SystemInfoResponse> {
  const { data } = await apiClient.get<
    ApiSuccessResponse<SystemInfoResponse>
  >("/system/info");
  if (!data.data) {
    throw new Error("System info not available");
  }
  return data.data;
}

export async function getSystemStatus(): Promise<SystemLiveStatus> {
  const { data } = await apiClient.get<
    ApiSuccessResponse<SystemLiveStatus>
  >("/system/status");
  if (!data.data) {
    throw new Error("System status not available");
  }
  return data.data;
}
