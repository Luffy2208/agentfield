import { Button } from "@/components/ui/button";
import { Copy, Check } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

function CopyableLine({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const short =
    value.length > 44 ? `${value.slice(0, 20)}…${value.slice(-12)}` : value;

  return (
    <div className="flex items-start justify-between gap-2 py-1.5 border-b border-border/60 last:border-0">
      <div className="min-w-0 flex-1">
        <p className="text-micro font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        <p
          className={cn(
            "mt-0.5 text-micro-plus text-foreground break-all",
            mono && "font-mono",
          )}
          title={value}
        >
          {short}
        </p>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-7 shrink-0 px-2 text-micro text-muted-foreground"
        onClick={() => {
          void navigator.clipboard.writeText(value).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? (
          <Check className="size-3" />
        ) : (
          <Copy className="size-3" />
        )}
      </Button>
    </div>
  );
}

/** Lean secondary card: VC-backed caller/target DIDs and content hashes (when DID/VC enabled). */
export function StepProvenanceCard({
  callerDid,
  targetDid,
  inputHash,
  outputHash,
}: {
  callerDid?: string;
  targetDid?: string;
  inputHash?: string;
  outputHash?: string;
}) {
  const rows: { label: string; value: string }[] = [];
  if (callerDid?.trim()) rows.push({ label: "Caller DID", value: callerDid.trim() });
  if (targetDid?.trim()) rows.push({ label: "Target DID", value: targetDid.trim() });
  if (inputHash?.trim()) rows.push({ label: "Input hash", value: inputHash.trim() });
  if (outputHash?.trim()) rows.push({ label: "Output hash", value: outputHash.trim() });

  if (rows.length === 0) return null;

  return (
    <div className="rounded-lg border border-border/80 bg-muted/30 px-3 py-2">
      <p className="text-micro font-medium text-muted-foreground mb-1">
        Provenance (VC)
      </p>
      <div className="divide-y-0">
        {rows.map((r) => (
          <CopyableLine key={r.label} label={r.label} value={r.value} mono />
        ))}
      </div>
    </div>
  );
}
