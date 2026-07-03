import { useQuery } from "@tanstack/react-query";
import { getPipelineStatus } from "../services/pipelineService";
import type { PipelineStatusResponse } from "../types/contract";

export function usePipelineStatus() {
  return useQuery<PipelineStatusResponse>({
    queryKey: ["pipeline", "status"],
    queryFn: getPipelineStatus,
    refetchInterval: 60000,
    staleTime: 30000,
  });
}
