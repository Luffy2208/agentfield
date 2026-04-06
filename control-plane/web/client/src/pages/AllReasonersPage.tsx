import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  SegmentedControl,
  type SegmentedControlOption,
} from "@/components/ui/segmented-control";
import {
  Grid,
  List,
  Renew,
  Terminal,
  Wifi,
  WifiOff,
} from "@/components/ui/icon-bridge";
import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { CompactReasonersStats } from "../components/reasoners/CompactReasonersStats";
import { PageHeader } from "../components/PageHeader";
import { EmptyReasonersState } from "../components/reasoners/EmptyReasonersState";
import { ReasonerGrid } from "../components/reasoners/ReasonerGrid";
import { SearchFilters } from "../components/reasoners/SearchFilters";
import { useSSESync } from "../hooks/useSSEQuerySync";
import { reasonersApi, ReasonersApiError } from "../services/reasonersApi";
import type {
  ReasonerFilters,
  ReasonersResponse,
  ReasonerWithNode,
} from "../types/reasoners";

const EMPTY_REASONERS: ReasonersResponse = {
  reasoners: [],
  total: 0,
  online_count: 0,
  offline_count: 0,
  nodes_count: 0,
};

type ViewMode = "grid" | "table";
const VIEW_OPTIONS: ReadonlyArray<SegmentedControlOption> = [
  { value: "grid", label: "Grid", icon: Grid },
  { value: "table", label: "Table", icon: List },
] as const;

export function AllReasonersPage() {
  const navigate = useNavigate();
  const { nodeConnected, reasonerConnected } = useSSESync();
  const reasonersLive = nodeConnected && reasonerConnected;
  const [filters, setFilters] = useState<ReasonerFilters>({
    status: "online",
    limit: 50,
    offset: 0,
  });
  const [viewMode, setViewMode] = useState<ViewMode>("grid");

  const reasonersQuery = useQuery({
    queryKey: [
      "reasoners",
      filters.status,
      filters.limit,
      filters.offset,
      filters.search ?? "",
    ],
    queryFn: async (): Promise<ReasonersResponse> => {
      try {
        return await reasonersApi.getAllReasoners(filters);
      } catch (err) {
        if (
          err instanceof ReasonersApiError &&
          (err.status === 404 ||
            (err.message && err.message.includes("no reasoners found")))
        ) {
          return { ...EMPTY_REASONERS };
        }
        throw err;
      }
    },
    placeholderData: keepPreviousData,
    refetchInterval: reasonersLive ? false : 6_000,
  });

  const {
    data,
    isPending,
    isFetching,
    isError,
    error: queryError,
    refetch,
    dataUpdatedAt,
  } = reasonersQuery;

  const errorMessage =
    isError && queryError instanceof Error
      ? queryError.message
      : isError
        ? "An unexpected error occurred while fetching reasoners"
        : null;

  const lastRefresh = useMemo(
    () => (dataUpdatedAt > 0 ? new Date(dataUpdatedAt) : new Date()),
    [dataUpdatedAt],
  );

  useEffect(() => {
    const handleClearFilters = () => {
      setFilters({
        status: "all",
        limit: 50,
        offset: 0,
      });
    };

    window.addEventListener("clearReasonerFilters", handleClearFilters);
    return () =>
      window.removeEventListener("clearReasonerFilters", handleClearFilters);
  }, []);

  const handleFiltersChange = (newFilters: ReasonerFilters) => {
    setFilters({ ...newFilters, offset: 0 });
  };

  const handleReasonerClick = (reasoner: ReasonerWithNode) => {
    navigate(`/reasoners/${encodeURIComponent(reasoner.reasoner_id)}`);
  };

  const handleRefresh = () => {
    void refetch();
  };

  const handleClearFilters = () => {
    setFilters({
      status: "online",
      limit: 50,
      offset: 0,
    });
  };

  const handleShowAll = () => {
    setFilters({
      status: "all",
      limit: 50,
      offset: 0,
    });
  };

  const getEmptyStateType = () => {
    if (!data) return null;

    if (data.total === 0) return "no-reasoners";
    if (filters.status === "online" && data.online_count === 0)
      return "no-online";
    if (filters.status === "offline" && data.offline_count === 0)
      return "no-offline";
    if (filters.search && data.reasoners.length === 0)
      return "no-search-results";
    if (data.reasoners.length === 0 && data.total > 0)
      return "no-search-results";

    return null;
  };

  const safeData = data ?? EMPTY_REASONERS;

  return (
    <div className="space-y-8">
      <PageHeader
        title="All Reasoners"
        description="Browse and execute reasoners across all connected agent nodes."
        aside={
          <div className="flex flex-wrap items-center gap-4">
            <SegmentedControl
              value={viewMode}
              onValueChange={(mode) => setViewMode(mode as ViewMode)}
              options={VIEW_OPTIONS}
              size="sm"
              optionClassName="min-w-[90px]"
              hideLabel
            />
            <Badge
              variant={reasonersLive ? "success" : "failed"}
              size="sm"
              showIcon={false}
              className="flex items-center gap-1"
            >
              {reasonersLive ? <Wifi size={12} /> : <WifiOff size={12} />}
              {reasonersLive ? "Live Updates" : "Disconnected"}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isFetching}
              className="flex items-center gap-2"
            >
              <Renew size={14} className={isFetching ? "animate-spin" : ""} />
              Refresh
            </Button>
          </div>
        }
      />

      <CompactReasonersStats
        total={safeData.total}
        onlineCount={safeData.online_count}
        offlineCount={safeData.offline_count}
        nodesCount={safeData.nodes_count}
        lastRefresh={lastRefresh}
        loading={isFetching}
        onRefresh={handleRefresh}
      />

      <SearchFilters
        filters={filters}
        onFiltersChange={handleFiltersChange}
        totalCount={safeData.total}
        onlineCount={safeData.online_count}
        offlineCount={safeData.offline_count}
      />

      {errorMessage ? (
        <Alert variant="destructive">
          <Terminal className="h-4 w-4" />
          <AlertTitle>Connection Error</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      ) : null}

      {!reasonersLive ? (
        <Alert>
          <WifiOff className="h-4 w-4" />
          <AlertTitle>Live updates unavailable</AlertTitle>
          <AlertDescription>
            One or more reasoner update streams are disconnected — this list
            polls every 6s until both reconnect. Use Refresh for an immediate
            pull.
          </AlertDescription>
        </Alert>
      ) : null}

      {(() => {
        const emptyStateType = getEmptyStateType();

        if (isPending && !data) {
          return (
            <ReasonerGrid
              reasoners={[]}
              loading={true}
              onReasonerClick={handleReasonerClick}
              viewMode={viewMode}
            />
          );
        }

        if (emptyStateType) {
          return (
            <EmptyReasonersState
              type={emptyStateType}
              searchTerm={filters.search}
              onRefresh={handleRefresh}
              onClearFilters={handleClearFilters}
              onShowAll={handleShowAll}
              loading={isFetching}
            />
          );
        }

        return (
          <ReasonerGrid
            reasoners={safeData.reasoners}
            loading={isFetching}
            onReasonerClick={handleReasonerClick}
            viewMode={viewMode}
          />
        );
      })()}

      {data && data.reasoners.length < data.total ? (
        <div className="flex justify-center mt-8">
          <Button
            variant="outline"
            onClick={() => {
              const newOffset = (filters.offset || 0) + (filters.limit || 50);
              setFilters({ ...filters, offset: newOffset });
            }}
            disabled={isFetching}
            className="flex items-center gap-2"
          >
            {isFetching ? (
              <>
                <Renew size={14} className="animate-spin" />
                Loading...
              </>
            ) : (
              <>
                Load More
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </>
            )}
          </Button>
        </div>
      ) : null}

      {!isFetching && !errorMessage && data && data.reasoners.length > 0 ? (
        <div className="text-center text-sm text-muted-foreground py-4">
          Last updated: {lastRefresh.toLocaleTimeString()}
        </div>
      ) : null}
    </div>
  );
}
