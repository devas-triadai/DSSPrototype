import type { EnemyAnalysis } from "../types";

interface EnemyAnalysisPanelProps {
  analysis: EnemyAnalysis | null | undefined;
}

export default function EnemyAnalysisPanel({
  analysis,
}: EnemyAnalysisPanelProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Enemy Analysis
      </h3>

      {!analysis ? (
        <p className="text-xs text-gray-600">No enemy analysis available</p>
      ) : (
        <div className="space-y-3">
          <StatusBadge
            label="Enemy Match"
            ok={analysis.enemy_match}
          />
          <div className="space-y-1.5 text-xs">
            <DataRow
              label="Confidence"
              value={`${(analysis.confidence * 100).toFixed(0)}%`}
            />
            {analysis.possible_equipment && (
              <DataRow
                label="Equipment"
                value={analysis.possible_equipment}
              />
            )}
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
          ? "bg-red-950/30 text-dss-danger"
          : "bg-green-950/30 text-dss-success"
      }`}
    >
      <span
        className={`h-2 w-2 rounded-full ${
          ok ? "bg-dss-danger" : "bg-dss-success"
        }`}
      />
      <span className="text-[11px] font-medium">
        {label}: {ok ? "Detected" : "None Detected"}
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
