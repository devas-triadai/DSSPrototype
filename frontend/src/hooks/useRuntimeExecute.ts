import { useMutation } from "@tanstack/react-query";
import { uploadAndExecute } from "../services/runtimeService";
import type { ImageMetadata, ExecuteResponse } from "../types";

interface ExecuteInput {
  file: File;
  metadata: ImageMetadata;
}

export function useRuntimeExecute() {
  return useMutation<ExecuteResponse, Error, ExecuteInput>({
    mutationFn: ({ file, metadata }: ExecuteInput) =>
      uploadAndExecute(file, metadata),
  });
}
