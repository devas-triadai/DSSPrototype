import type { FusionResult } from "../types";

interface FusionSummaryProps {
  fusion: FusionResult | null | undefined;
}

export default function FusionSummary({ fusion }: FusionSummaryProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Fusion Summary
      </h3>

      {!fusion ? (
        <p className="text-xs text-gray-600">No fusion data available</p>
      ) : (
        <div className="space-y-2.5">
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-gray-500">Confidence</span>
            <div className="flex items-center gap-2">
              <div className="h-2 w-24 overflow-hidden rounded-full bg-gray-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    fusion.combined_confidence >= 0.7
                      ? "bg-dss-success"
                      : fusion.combined_confidence >= 0.4
                        ? "bg-dss-accent"
                        : "bg-dss-danger"
                  }`}
                  style={{
                    width: `${(fusion.combined_confidence * 100).toFixed(0)}%`,
                  }}
                />
              </div>
              <span className="font-mono text-[11px] text-gray-400">
                {(fusion.combined_confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div>
            <p className="mb-0.5 text-[10px] text-gray-500">Summary</p>
            <p className="text-xs leading-relaxed text-gray-200">
              {fusion.summary}
            </p>
          </div>

          {fusion.supporting_evidence.length > 0 && (
            <div>
              <p className="mb-1 text-[10px] text-gray-500">
                Supporting Evidence
              </p>
              <ul className="space-y-0.5">
                {fusion.supporting_evidence.map((ev, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-1.5 text-[11px] text-gray-400"
                  >
                    <span className="mt-0.5 text-dss-accent">&bull;</span>
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
