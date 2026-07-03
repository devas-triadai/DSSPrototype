import type { TerrainAnalysis } from "../types";
import { TerrainType } from "../types";

interface TerrainAnalysisPanelProps {
  analysis: TerrainAnalysis | null | undefined;
}

const terrainLabels: Record<TerrainType, string> = {
  [TerrainType.ROAD]: "Road",
  [TerrainType.FOREST]: "Forest",
  [TerrainType.RIVER]: "River",
  [TerrainType.HILL]: "Hill",
  [TerrainType.OPEN_FIELD]: "Open Field",
  [TerrainType.URBAN]: "Urban",
  [TerrainType.BRIDGE]: "Bridge",
  [TerrainType.UNKNOWN]: "Unknown",
};

export default function TerrainAnalysisPanel({
  analysis,
}: TerrainAnalysisPanelProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Terrain Analysis
      </h3>

      {!analysis ? (
        <p className="text-xs text-gray-600">No terrain analysis available</p>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="rounded bg-gray-800 px-2 py-0.5 font-mono text-[11px] uppercase text-dss-info">
              {terrainLabels[analysis.terrain_type] || analysis.terrain_type}
            </span>
          </div>

          <div className="space-y-1.5 text-xs">
            <DataRow
              label="Visibility"
              value={analysis.visibility}
            />
            <DataRow
              label="Road Access"
              value={analysis.road_access ? "Yes" : "No"}
            />
            {analysis.elevation !== null && analysis.elevation !== undefined && (
              <DataRow
                label="Elevation"
                value={`${analysis.elevation.toFixed(0)}m`}
              />
            )}
            {analysis.nearby_features.length > 0 && (
              <div>
                <p className="mb-0.5 text-[10px] text-gray-500">
                  Nearby Features
                </p>
                <div className="flex flex-wrap gap-1">
                  {analysis.nearby_features.map((f, i) => (
                    <span
                      key={i}
                      className="rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
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

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono text-gray-300">{value}</span>
    </div>
  );
}
