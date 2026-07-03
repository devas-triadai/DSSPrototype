import { useQuery } from "@tanstack/react-query";
import { getDecisionCapabilities } from "../services/decisionService";
import type { DecisionCapabilitiesResponse } from "../types/contract";

export function useDecisionCapabilities() {
  return useQuery<DecisionCapabilitiesResponse>({
    queryKey: ["decision", "capabilities"],
    queryFn: getDecisionCapabilities,
    staleTime: 120000,
  });
}
