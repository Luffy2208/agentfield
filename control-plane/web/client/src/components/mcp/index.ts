// MCP component stubs for NodeDetailPage

interface MCPServerListProps {
  servers: unknown[];
  nodeId: string;
  onServerAction?: (action: string, serverAlias: string) => Promise<void>;
}

interface MCPServerControlsProps {
  servers: unknown[];
  nodeId: string;
  onBulkAction?: (action: string, serverAliases: string[]) => Promise<void>;
}

interface MCPToolExplorerProps {
  tools: unknown[];
  serverAlias: string;
  nodeId: string;
}

interface MCPToolTesterProps {
  tool: {
    name: string;
    description: string;
    input_schema: { type: string; properties: Record<string, unknown> };
  };
  serverAlias: string;
  nodeId: string;
}

export function MCPServerList(_props: MCPServerListProps) {
  return null;
}

export function MCPServerControls(_props: MCPServerControlsProps) {
  return null;
}

export function MCPToolExplorer(_props: MCPToolExplorerProps) {
  return null;
}

export function MCPToolTester(_props: MCPToolTesterProps) {
  return null;
}
