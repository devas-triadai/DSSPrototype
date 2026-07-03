import { useState, useCallback } from "react";
import type { ExecuteResponse } from "../types";
import type { ImageMetadata } from "../types";
import { useRuntimeExecute } from "../hooks";
import SystemStatusBar from "../components/SystemStatusBar";
import PipelineVisualization from "../components/PipelineVisualization";
import ImageUploadPanel from "../components/ImageUploadPanel";
import ImagePreview from "../components/ImagePreview";
import UploadHistory from "../components/UploadHistory";
import DetectionResults from "../components/DetectionResults";
import FriendlyAnalysisPanel from "../components/FriendlyAnalysisPanel";
import EnemyAnalysisPanel from "../components/EnemyAnalysisPanel";
import TerrainAnalysisPanel from "../components/TerrainAnalysisPanel";
import FusionSummary from "../components/FusionSummary";
import ThreatAssessmentPanel from "../components/ThreatAssessmentPanel";
import DecisionRecommendationPanel from "../components/DecisionRecommendationPanel";
import ExecutionMetrics from "../components/ExecutionMetrics";
import ErrorDisplay from "../components/ErrorDisplay";

interface UploadEntry {
  imageId: string;
  filename: string;
  timestamp: string;
  status: string;
}

export default function MissionDashboard() {
  const [executionResult, setExecutionResult] =
    useState<ExecuteResponse | null>(null);
  const [uploadHistory, setUploadHistory] = useState<UploadEntry[]>([]);
  const [currentMetadata, setCurrentMetadata] =
    useState<ImageMetadata | null>(null);

  const executeMutation = useRuntimeExecute();

  const handleExecute = useCallback(
    (file: File, metadata: ImageMetadata) => {
      setCurrentMetadata(metadata);
      executeMutation.mutate(
        { file, metadata },
        {
          onSuccess: (data) => {
            setExecutionResult(data);
            setUploadHistory((prev) => [
              {
                imageId: metadata.image_id,
                filename: file.name,
                timestamp: new Date().toISOString(),
                status: data.status,
              },
              ...prev,
            ]);
          },
          onError: () => {
            setUploadHistory((prev) => [
              {
                imageId: metadata.image_id,
                filename: file.name,
                timestamp: new Date().toISOString(),
                status: "failed",
              },
              ...prev,
            ]);
          },
        },
      );
    },
    [executeMutation],
  );

  const handleHistorySelect = useCallback(
    (imageId: string) => {
      const entry = uploadHistory.find((e) => e.imageId === imageId);
      if (entry) {
        setCurrentMetadata((prev) =>
          prev ? { ...prev, image_id: imageId } : null,
        );
      }
    },
    [uploadHistory],
  );

  const isRunning = executeMutation.isPending;
  const hasResult = executionResult !== null;

  return (
    <div className="flex flex-col">
      <SystemStatusBar />

      <PipelineVisualization
        result={executionResult}
        isRunning={isRunning}
      />

      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-12">
        {/* Left sidebar: upload, preview, history */}
        <aside className="space-y-4 lg:col-span-3">
          <ImageUploadPanel
            onExecute={handleExecute}
            isRunning={isRunning}
          />
          {currentMetadata && (
            <ImagePreview metadata={currentMetadata} />
          )}
          <UploadHistory
            entries={uploadHistory}
            onSelect={handleHistorySelect}
          />
        </aside>

        {/* Center: detection results */}
        <div className="space-y-4 lg:col-span-5">
          <DetectionResults
            detection={executionResult?.detection ?? null}
            isLoading={isRunning}
          />
        </div>

        {/* Right sidebar: analysis panels */}
        <aside className="space-y-4 lg:col-span-4">
          <FriendlyAnalysisPanel
            analysis={executionResult?.friendly ?? null}
          />
          <EnemyAnalysisPanel
            analysis={executionResult?.enemy ?? null}
          />
          <TerrainAnalysisPanel
            analysis={executionResult?.terrain ?? null}
          />
        </aside>
      </div>

      {/* Error display */}
      {executeMutation.isError && (
        <div className="px-4 pb-4">
          <ErrorDisplay message={executeMutation.error.message} />
        </div>
      )}

      {/* Bottom row: fusion, threat, decision, metrics */}
      {hasResult && (
        <div className="grid grid-cols-1 gap-4 px-4 pb-6 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <FusionSummary fusion={executionResult?.fusion ?? null} />
          </div>
          <div className="lg:col-span-3">
            <ThreatAssessmentPanel
              threat={executionResult?.threat ?? null}
            />
          </div>
          <div className="lg:col-span-5">
            <DecisionRecommendationPanel
              decision={executionResult?.decision ?? null}
            />
          </div>
          <div className="lg:col-span-12">
            <ExecutionMetrics result={executionResult} />
          </div>
        </div>
      )}
    </div>
  );
}
