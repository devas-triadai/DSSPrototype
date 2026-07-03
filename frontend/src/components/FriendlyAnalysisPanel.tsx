import type { FriendlyAnalysis } from "../types";

interface FriendlyAnalysisPanelProps {
  analysis: FriendlyAnalysis | null | undefined;
}

export default function FriendlyAnalysisPanel({
  analysis,
}: FriendlyAnalysisPanelProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Friendly Analysis
      </h3>

      {!analysis ? (
        <p className="text-xs text-gray-600">
          No friendly analysis available
        </p>
      ) : (
        <div className="space-y-3">
          <StatusBadge
            label="Friendly Match"
            ok={analysis.friendly_match}
          />
          <div className="space-y-1.5 text-xs">
            <DataRow
              label="Confidence"
              value={`${(analysis.confidence * 100).toFixed(0)}%`}
            />
            <div>
              <p className="mb-0.5 text-[10px] text-gray-500">Assessment</p>
              <p className="text-xs leading-relaxed text-gray-300">
                {analysis.reason}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div
      className={`flex items-center gap-2 rounded px-2.5 py-1.5 ${
        ok
          ? "bg-green-950/30 text-dss-success"
          : "bg-amber-950/30 text-amber-400"
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${
          ok ? "bg-dss-success" : "bg-amber-400"
        }`}
      />
      <span className="text-[11px] font-medium">
        {label}: {ok ? "Match" : "No Match"}
      </span>
    </div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono text-gray-300">{value}</span>
    </div>
  );
}
