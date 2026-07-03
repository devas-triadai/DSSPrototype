import { useQuery } from "@tanstack/react-query";
import { getSystemStatus } from "../services/systemService";
import type { SystemLiveStatus } from "../types/api";

export function useSystemStatus() {
  return useQuery<SystemLiveStatus>({
    queryKey: ["system", "status"],
    queryFn: getSystemStatus,
    refetchInterval: 30000,
    staleTime: 15000,
  });
}
