import apiClient from "./api";
import type {
  ImageMetadata,
  VisionValidateResponse,
  ApiSuccessResponse,
} from "../types";

export async function validateImage(
  metadata: ImageMetadata,
): Promise<VisionValidateResponse> {
  const { data } = await apiClient.post<
    ApiSuccessResponse<VisionValidateResponse>
  >("/vision/validate", metadata);
  if (!data.data) {
    throw new Error("Validation returned no data");
  }
  return data.data;
}
