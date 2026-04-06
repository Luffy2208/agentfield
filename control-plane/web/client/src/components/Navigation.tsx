import { Activity, Layers, Network_3, Package, Settings, Dashboard } from "@/components/ui/icon-bridge";
import { useMode } from "../contexts/ModeContext";

interface NavigationProps {
  currentPage: string;
  onPageChange: (page: string) => void;
}

export function Navigation({ currentPage, onPageChange }: NavigationProps) {
  const { mode } = useMode();

  // Developer mode navigation items
  const developerItems = [
    {
      id: "nodes",
      label: "Agent Nodes",
      icon: Network_3,
      description: "Monitor and manage AI agent nodes",
      disabled: false,
    },
    {
      id: "executions",
      label: "Executions",
      icon: Activity,
      description: "Track workflow executions and performance",
      disabled: false,
    },
    {
      id: "workflows",
      label: "Workflows",
      icon: Layers,
      description: "Manage workflow DAGs and dependencies",
      disabled: true,
    },
    {
      id: "packages",
      label: "Packages",
      icon: Package,
      description: "Manage agent packages and configurations",
      disabled: false,
    },
    {
      id: "settings",
      label: "Settings",
      icon: Settings,
      description: "System configuration and preferences",
      disabled: false,
    },
  ];

  // User mode navigation items
  const userItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: Dashboard,
      description: "Overview of your agents and activities",
      disabled: false,
    },
    {
      id: "agents",
      label: "Agent nodes",
      icon: Network_3,
      description: "Registered nodes, endpoints, and process logs",
      disabled: false,
    },
    {
      id: "packages",
      label: "Agent Store",
      icon: Package,
      description: "Browse and install agent packages",
      disabled: false,
    },
    {
      id: "executions",
      label: "Activity",
      icon: Activity,
      description: "Recent agent activities and results",
      disabled: false,
    },
  ];

  const navigationItems = mode === 'developer' ? developerItems : userItems;

  return (
    <nav className="flex items-center gap-0.5 bg-muted p-1 rounded-lg border border-border">
      {navigationItems.map((item) => {
        const Icon = item.icon;
        const isActive = currentPage === item.id;

        return (
          <button
            key={item.id}
            onClick={() => !item.disabled && onPageChange(item.id)}
            disabled={item.disabled}
            className={`
              relative flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium
              transition-all duration-200 ease-in-out group
              ${
                isActive
                  ? "bg-card text-foreground shadow-sm border border-border"
                  : item.disabled
                  ? "text-muted-foreground opacity-50 cursor-not-allowed opacity-50"
                  : "text-muted-foreground hover:text-foreground hover:bg-card/60"
              }
              ${!item.disabled ? "cursor-pointer" : ""}
            `}
            title={item.description}
          >
            <Icon className={`h-4 w-4 transition-transform duration-200 ${!item.disabled ? 'group-hover:scale-105' : ''}`} />
            <span className="whitespace-nowrap hidden sm:inline">{item.label}</span>

            {/* Active indicator */}
            {isActive && (
              <div className="absolute inset-x-0 bottom-0 h-0.5 bg-foreground rounded-full" />
            )}
          </button>
        );
      })}
    </nav>
  );
}
