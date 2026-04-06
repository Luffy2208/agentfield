/**
 * MCP (Model Context Protocol) Module Export
 *
 * NOTE: MCP UI components have been removed. This file retains exports for
 * API services and types that may still be used by the control plane layer.
 */

// ============================================================================
// MCP API Services
// ============================================================================
export {
  getMCPHealth,
  getMCPHealthModeAware,
  restartMCPServer,
  stopMCPServer,
  startMCPServer,
  getMCPTools,
  getOverallMCPStatus,
  getNodeDetailsWithMCP,
  testMCPTool,
  getMCPServerMetrics,
  subscribeMCPHealthEvents,
  getMCPHealthEvents,
  bulkMCPServerAction,
  getMCPServerConfig,
  updateMCPServerConfig
} from '../services/api';

// ============================================================================
// MCP Types
// ============================================================================
export type {
  MCPServerAction,
  MCPSummaryForUI,
  MCPServerHealthForUI,
  MCPTool,
  MCPToolTestRequest,
  MCPToolTestResponse,
  MCPHealthEvent,
  MCPServerMetrics,
  MCPNodeMetrics,
  MCPErrorDetails,
  MCPError,
  AgentNodeDetailsForUI,
  MCPHealthResponse,
  MCPServerActionResponse,
  MCPToolsResponse,
  MCPOverallStatusResponse,
  MCPServerMetricsResponse,
  MCPHealthEventResponse,
  MCPHealthResponseModeAware,
  MCPHealthResponseUser,
  MCPHealthResponseDeveloper,
  AppMode
} from '../types/agentfield';
