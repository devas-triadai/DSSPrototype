import { useMutation } from "@tanstack/react-query";
import { validateImage } from "../services/visionService";
import type { ImageMetadata, VisionValidateResponse } from "../types";

export function useVisionValidate() {
  return useMutation<VisionValidateResponse, Error, ImageMetadata>({
    mutationFn: validateImage,
  });
}
