import type { ExecuteResponse } from "../types";

interface ExecutionMetricsProps {
  result: ExecuteResponse | null;
}

const stageLabels: Record<string, string> = {
  computer_vision: "Computer Vision",
  friendly: "Friendly Knowledge",
  enemy: "Enemy Knowledge",
  terrain: "Terrain Knowledge",
  fusion: "Fusion Engine",
  decision: "Decision Engine",
};

export default function ExecutionMetrics({
  result,
}: ExecutionMetricsProps) {
  if (!result) {
    return (
      <div className="rounded-lg border border-dss-border bg-dss-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
          Execution Metrics
        </h3>
        <p className="text-xs text-gray-600">
          Run pipeline to see metrics
        </p>
      </div>
    );
  }

  const stages = result.stage_durations;
  const maxDuration = Math.max(...Object.values(stages), 1);

  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Execution Metrics
      </h3>

      <div className="mb-3 flex gap-4 border-b border-dss-border pb-2 text-xs">
        <div>
          <p className="text-[10px] text-gray-500">Total</p>
          <p className="font-mono text-sm font-bold text-gray-200">
            {result.total_duration_ms.toFixed(1)}ms
          </p>
        </div>
        <div>
          <p className="text-[10px] text-gray-500">Status</p>
          <p
            className={`font-mono text-sm font-bold ${
              result.status === "completed"
                ? "text-dss-success"
                : "text-dss-danger"
            }`}
          >
            {result.status}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-gray-500">Pipeline</p>
          <p className="font-mono text-xs text-gray-400">
            {result.pipeline_id.slice(0, 12)}
          </p>
        </div>
      </div>

      <div className="space-y-1.5">
        {Object.entries(stages).map(([key, duration]) => {
          const label = stageLabels[key] || key;
          const pct = (duration / maxDuration) * 100;
          return (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="w-28 shrink-0 text-right text-[10px] text-gray-500">
                {label}
              </span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-800">
                <div
                  className="h-full rounded-full bg-dss-info transition-all duration-500"
                  style={{ width: `${Math.max(pct, 2)}%` }}
                />
              </div>
              <span className="w-14 text-right font-mono text-[10px] text-gray-400">
                {duration.toFixed(1)}ms
              </span>
            </div>
          );
        })}
      </div>

      {result.errors.length > 0 && (
        <div className="mt-3 space-y-1">
          <p className="text-[10px] font-medium text-dss-danger">Errors</p>
          {result.errors.map((err, i) => (
            <p key={i} className="text-[11px] text-red-400">
              {err}
            </p>
          ))}
        </div>
      )}

      {result.warnings.length > 0 && (
        <div className="mt-2 space-y-1">
          <p className="text-[10px] font-medium text-amber-400">Warnings</p>
          {result.warnings.map((warn, i) => (
            <p key={i} className="text-[11px] text-amber-400/70">
              {warn}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
