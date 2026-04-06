import { useQuery } from "@tanstack/react-query";
import { getNodesSummary } from "../../services/api";
import type { AgentNodeSummary } from "../../types/agentfield";
import { useSSESync } from "../useSSEQuerySync";

interface NodesSummaryResponse {
  nodes: AgentNodeSummary[];
  count: number;
}

export function useAgents() {
  const { nodeConnected } = useSSESync();
  return useQuery<NodesSummaryResponse>({
    queryKey: ["agents"],
    queryFn: () => getNodesSummary(),
    refetchInterval: nodeConnected ? 10_000 : 5_000,
  });
}
