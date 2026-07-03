import { useHealth, useSystemStatus, usePipelineStatus } from "../hooks";

export default function SystemStatusBar() {
  const { data: health, isLoading: healthLoading } = useHealth();
  const { data: status, isLoading: statusLoading } = useSystemStatus();
  const { data: pipeline } = usePipelineStatus();

  const loading = healthLoading || statusLoading;

  if (loading) {
    return (
      <div className="border-b border-dss-border bg-dss-panel/80 px-6 py-2">
        <div className="h-5 w-64 animate-pulse rounded bg-gray-700" />
      </div>
    );
  }

  return (
    <div className="border-b border-dss-border bg-dss-panel/80 px-6 py-2">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs font-medium">
        <StatusDot
          label="API"
          healthy={health?.status === "healthy"}
        />
        <StatusDot
          label="Runtime"
          healthy={status?.healthy ?? false}
        />
        <div className="flex items-center gap-1.5 text-dss-muted">
          <span>Modules</span>
          <span className="font-mono text-gray-300">
            {status?.modules_registered ?? "?"}/6
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-dss-muted">
          <span>Pipeline</span>
          <span
            className={
              pipeline?.status === "configured"
                ? "text-dss-success"
                : "text-dss-danger"
            }
          >
            {pipeline?.status ?? "unknown"}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-dss-muted">
          <span>Stages</span>
          <span className="font-mono text-gray-300">
            {pipeline?.stage_count ?? "?"}
          </span>
        </div>
        {status?.missing_modules && status.missing_modules.length > 0 && (
          <div className="flex items-center gap-1.5 text-amber-400">
            <span>Missing</span>
            <span className="font-mono">
              {status.missing_modules.join(", ")}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusDot({
  label,
  healthy,
}: {
  label: string;
  healthy: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={`inline-block h-2 w-2 rounded-full ${
          healthy ? "bg-dss-success shadow-sm shadow-dss-success/50" : "bg-dss-danger"
        }`}
      />
      <span className={healthy ? "text-gray-300" : "text-dss-danger"}>
        {label}
      </span>
      <span className="text-dss-muted">{healthy ? "Healthy" : "Down"}</span>
    </div>
  );
}
