import type { ThreatAssessment } from "../types";
import { ThreatLevel } from "../types";

interface ThreatAssessmentPanelProps {
  threat: ThreatAssessment | null | undefined;
}

const threatColors: Record<ThreatLevel, { bg: string; text: string; bar: string }> = {
  [ThreatLevel.CRITICAL]: {
    bg: "bg-red-950/40",
    text: "text-red-400",
    bar: "bg-dss-danger",
  },
  [ThreatLevel.HIGH]: {
    bg: "bg-orange-950/40",
    text: "text-orange-400",
    bar: "bg-orange-500",
  },
  [ThreatLevel.MEDIUM]: {
    bg: "bg-amber-950/30",
    text: "text-amber-400",
    bar: "bg-dss-accent",
  },
  [ThreatLevel.LOW]: {
    bg: "bg-green-950/30",
    text: "text-green-400",
    bar: "bg-dss-success",
  },
  [ThreatLevel.UNKNOWN]: {
    bg: "bg-gray-800/50",
    text: "text-gray-500",
    bar: "bg-gray-600",
  },
};

export default function ThreatAssessmentPanel({
  threat,
}: ThreatAssessmentPanelProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Threat Assessment
      </h3>

      {!threat ? (
        <p className="text-xs text-gray-600">No threat assessment available</p>
      ) : (
        <div className="space-y-3">
          <div
            className={`flex items-center gap-2 rounded px-3 py-2 ${
              threatColors[threat.threat_level]?.bg || "bg-gray-800/50"
            }`}
          >
            <span
              className={`text-sm font-bold uppercase ${
                threatColors[threat.threat_level]?.text || "text-gray-500"
              }`}
            >
              {threat.threat_level}
            </span>
          </div>

          <div className="space-y-1.5 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Confidence</span>
              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-gray-800">
                <div
                  className={`h-full rounded-full ${
                    threatColors[threat.threat_level]?.bar || "bg-gray-600"
                  }`}
                  style={{
                    width: `${(threat.confidence * 100).toFixed(0)}%`,
                  }}
                />
              </div>
              <span className="font-mono text-[10px] text-gray-400">
                {(threat.confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div>
              <p className="mb-0.5 text-[10px] text-gray-500">Assessment</p>
              <p className="text-xs leading-relaxed text-gray-300">
                {threat.reason}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
