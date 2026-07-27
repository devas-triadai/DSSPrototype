import type { DetectionResult } from "../types";
import BoundingBoxInfo from "./BoundingBoxInfo";

interface DetectionResultsProps {
  detection: DetectionResult | null | undefined;
  isLoading: boolean;
}

export default function DetectionResults({
  detection,
  isLoading,
}: DetectionResultsProps) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-dss-border bg-dss-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
          Detection Results
        </h3>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-6 w-full animate-pulse rounded bg-gray-800"
            />
          ))}
        </div>
      </div>
    );
  }

  if (!detection) {
    return (
      <div className="rounded-lg border border-dss-border bg-dss-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
          Detection Results
        </h3>
        <p className="text-xs text-gray-600">
          Run pipeline execution to see results
        </p>
      </div>
    );
  }

  const objects = detection.objects;

  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Detection Results
      </h3>

      <div className="mb-2 flex gap-4 text-[11px] text-gray-400">
        <span>
          Model:{" "}
          <span className="font-mono text-gray-300">
            {detection.model_version}
          </span>
        </span>
        <span>
          Objects:{" "}
          <span className="font-mono text-gray-300">{objects.length}</span>
        </span>
        <span>
          Time:{" "}
          <span className="font-mono text-gray-300">
            {detection.processing_time_ms.toFixed(1)}ms
          </span>
        </span>
      </div>

      {objects.length === 0 ? (
        <p className="text-xs text-gray-500">No objects detected</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-dss-border text-left text-gray-500">
                  <th className="pb-1.5 pr-2 font-medium">Type</th>
                  <th className="pb-1.5 pr-2 font-medium">Label</th>
                  <th className="pb-1.5 pr-2 font-medium">Confidence</th>
                  <th className="pb-1.5 font-medium">Position</th>
                </tr>
              </thead>
              <tbody className="text-gray-300">
                {objects.map((obj) => (
                  <tr
                    key={obj.id}
                    className="border-b border-dss-border/30 hover:bg-gray-800/30"
                  >
                    <td className="py-1.5 pr-2">
                      <span className="rounded bg-gray-800 px-1.5 py-0.5 font-mono text-[10px] uppercase text-dss-info">
                        {obj.object_type.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td className="py-1.5 pr-2 text-gray-400">
                      {obj.name || "\u2014"}
                    </td>
                    <td className="py-1.5 pr-2">
                      <ConfidenceBar value={obj.confidence} />
                    </td>
                    <td className="py-1.5 font-mono text-[10px] text-gray-500">
                      ({obj.geometry.box.x.toFixed(0)},{" "}
                      {obj.geometry.box.y.toFixed(0)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <BoundingBoxInfo objects={objects} />
        </>
      )}
    </div>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = (value * 100).toFixed(0);
  const color =
    value >= 0.8
      ? "bg-dss-success"
      : value >= 0.5
        ? "bg-dss-accent"
        : "bg-dss-danger";

  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-gray-800">
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-gray-400">{pct}%</span>
    </div>
  );
}
