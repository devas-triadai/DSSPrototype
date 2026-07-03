import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../services/healthService";
import type { HealthResponse } from "../types/api";

export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30000,
    staleTime: 15000,
  });
}
