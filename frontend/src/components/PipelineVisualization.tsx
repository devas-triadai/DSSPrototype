import type { ExecuteResponse } from "../types";

interface PipelineVisualizationProps {
  result: ExecuteResponse | null;
  isRunning: boolean;
}

interface StageDef {
  key: string;
  label: string;
}

const STAGES: StageDef[] = [
  { key: "computer_vision", label: "Computer Vision" },
  { key: "friendly", label: "Friendly" },
  { key: "enemy", label: "Enemy" },
  { key: "terrain", label: "Terrain" },
  { key: "fusion", label: "Fusion" },
  { key: "decision", label: "Decision" },
];

type StageStatus = "idle" | "pending" | "running" | "completed" | "failed";

function getStageStatus(
  key: string,
  result: ExecuteResponse | null,
  isRunning: boolean,
): StageStatus {
  if (!result) return isRunning ? "pending" : "idle";

  if (result.status === "failed") {
    if (result.detection && key === "computer_vision") return "completed";
    if (result.friendly && key === "friendly") return "completed";
    if (result.enemy && key === "enemy") return "completed";
    if (result.terrain && key === "terrain") return "completed";
    if (result.fusion && key === "fusion") return "completed";
    if (result.threat && (key === "fusion" || key === "decision"))
      return "completed";
    if (result.decision && key === "decision") return "completed";
    return "failed";
  }

  if (result.status === "completed") {
    const stageMap: Record<string, boolean> = {
      computer_vision: !!result.detection,
      friendly: !!result.friendly,
      enemy: !!result.enemy,
      terrain: !!result.terrain,
      fusion: !!result.fusion,
      decision: !!result.decision,
    };
    return stageMap[key] ? "completed" : "failed";
  }

  return "idle";
}

function stageIndicator(status: StageStatus): string {
  switch (status) {
    case "completed":
      return "bg-dss-success";
    case "failed":
      return "bg-dss-danger";
    case "running":
      return "bg-dss-accent animate-pulse";
    case "pending":
      return "bg-gray-600";
    default:
      return "bg-gray-700";
  }
}

function stageLabel(status: StageStatus): string {
  switch (status) {
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "running":
      return "Running\u2026";
    case "pending":
      return "Pending";
    default:
      return "Idle";
  }
}

export default function PipelineVisualization({
  result,
  isRunning,
}: PipelineVisualizationProps) {
  return (
    <div className="border-b border-dss-border bg-dss-panel/60 px-6 py-3">
      <div className="flex items-center justify-center gap-0">
        {STAGES.map((stage, idx) => {
          const status = getStageStatus(stage.key, result, isRunning);
          return (
            <div key={stage.key} className="flex items-center">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full ${stageIndicator(status)} transition-colors duration-300`}
                >
                  <span className="text-[10px] font-bold text-gray-950">
                    {idx + 1}
                  </span>
                </div>
                <span className="text-[10px] font-medium text-gray-400">
                  {stage.label}
                </span>
                <span
                  className={`text-[9px] font-mono uppercase ${
                    status === "completed"
                      ? "text-dss-success"
                      : status === "failed"
                        ? "text-dss-danger"
                        : "text-gray-600"
                  }`}
                >
                  {stageLabel(status)}
                </span>
              </div>
              {idx < STAGES.length - 1 && (
                <div className="mx-1 mt-[-20px]">
                  <Arrow right />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Arrow({ right }: { right: boolean }) {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      className={`text-gray-600 ${right ? "" : "rotate-180"}`}
    >
      <path
        d="M9 6l6 6-6 6"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
