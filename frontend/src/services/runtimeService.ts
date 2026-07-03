import apiClient from "./api";
import type {
  ImageMetadata,
  ExecuteResponse,
  ApiSuccessResponse,
} from "../types";

export async function executePipeline(
  metadata: ImageMetadata,
): Promise<ExecuteResponse> {
  const { data } = await apiClient.post<
    ApiSuccessResponse<ExecuteResponse>
  >("/runtime/execute", metadata);
  if (!data.data) {
    throw new Error("Pipeline execution returned no data");
  }
  return data.data;
}

export async function uploadAndExecute(
  file: File,
  metadata: ImageMetadata,
): Promise<ExecuteResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("image_id", metadata.image_id);
  formData.append("timestamp", metadata.timestamp);
  formData.append("source", metadata.source ?? "local_upload");
  if (metadata.width !== undefined) {
    formData.append("width", String(metadata.width));
  }
  if (metadata.height !== undefined) {
    formData.append("height", String(metadata.height));
  }
  if (metadata.format !== undefined) {
    formData.append("format", metadata.format);
  }

  const { data } = await apiClient.post<
    ApiSuccessResponse<ExecuteResponse>
  >("/runtime/upload", formData);
  if (!data.data) {
    throw new Error("Pipeline execution returned no data");
  }
  return data.data;
}
