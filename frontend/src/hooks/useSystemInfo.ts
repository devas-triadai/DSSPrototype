import { useQuery } from "@tanstack/react-query";
import { getSystemInfo } from "../services/systemService";
import type { SystemInfoResponse } from "../types/contract";

export function useSystemInfo() {
  return useQuery<SystemInfoResponse>({
    queryKey: ["system", "info"],
    queryFn: getSystemInfo,
    refetchInterval: 30000,
    staleTime: 15000,
  });
}
