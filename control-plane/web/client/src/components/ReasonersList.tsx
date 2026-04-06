import React from 'react';
import type { ReasonerDefinition } from '../types/agentfield';
import { Badge } from '@/components/ui/badge';
import { ReasonerIcon } from '@/components/ui/icon-bridge';

interface ReasonersListProps {
  reasoners: ReasonerDefinition[];
}

const ReasonersList: React.FC<ReasonersListProps> = ({ reasoners }) => {
  if (!reasoners || reasoners.length === 0) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <ReasonerIcon className="h-4 w-4 text-muted-foreground" />
          <h4 className="text-sm font-medium">Reasoners (0)</h4>
        </div>
        <p className="text-sm text-muted-foreground">No reasoners available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ReasonerIcon className="h-4 w-4 text-muted-foreground" />
        <h4 className="text-sm font-medium">Reasoners ({reasoners.length})</h4>
      </div>
      <div className="flex flex-wrap gap-2">
        {reasoners.map((reasoner) => (
          <div
            key={reasoner.id}
            className="min-w-[140px] rounded-lg border border-border bg-card px-3 py-2"
          >
            <div className="text-xs font-medium text-foreground">
              {reasoner.id}
            </div>
            {reasoner.tags && reasoner.tags.length > 0 ? (
              <div className="mt-1 flex flex-wrap gap-1">
                {reasoner.tags.slice(0, 3).map((tag) => (
                  <Badge
                    key={`${reasoner.id}-${tag}`}
                    variant="outline"
                    className="text-micro bg-background text-muted-foreground border-border"
                  >
                    #{tag}
                  </Badge>
                ))}
                {reasoner.tags.length > 3 && (
                  <Badge
                    variant="outline"
                    className="text-micro bg-background text-muted-foreground border-border"
                  >
                    +{reasoner.tags.length - 3}
                  </Badge>
                )}
              </div>
            ) : (
              <p className="mt-1 text-micro-plus text-muted-foreground">No tags</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReasonersList;
