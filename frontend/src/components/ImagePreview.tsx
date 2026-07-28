import { useRef, useState, useEffect } from "react";
import type { ImageMetadata, DetectedObject } from "../types";

interface ImagePreviewProps {
  metadata: ImageMetadata;
  previewUrl: string | null;
  objects: DetectedObject[] | null;
}

const BOX_COLORS = [
  "#f59e0b",
  "#10b981",
  "#3b82f6",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
];

export default function ImagePreview({
  metadata,
  previewUrl,
  objects,
}: ImagePreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!previewUrl) {
      setImgSize(null);
      return;
    }
    const img = new Image();
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = previewUrl;
  }, [previewUrl]);

  const hasData = metadata.image_id.trim().length > 0;

  if (!hasData) {
    return (
      <div className="rounded-lg border border-dss-border bg-dss-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
          Image Preview
        </h3>
        <div className="flex aspect-video items-center justify-center rounded border border-dashed border-gray-700 bg-gray-900/50">
          <p className="text-xs text-gray-600">No image loaded</p>
        </div>
      </div>
    );
  }

  const dimensionsLabel = [metadata.width, metadata.height]
    .filter((d): d is number => d !== undefined && d !== null)
    .join("x");

  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Image Preview
      </h3>

      <div
        ref={containerRef}
        className="relative overflow-hidden rounded border border-dss-border bg-gray-900/50"
      >
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={metadata.image_id}
            className="block w-full object-contain"
          />
        ) : (
          <div className="flex aspect-video items-center justify-center">
            <div className="text-center">
              <div className="text-3xl text-gray-700">
                <SvgPlaceholder />
              </div>
              <p className="mt-1 font-mono text-[11px] text-gray-600">
                {metadata.image_id}
              </p>
            </div>
          </div>
        )}

        {previewUrl &&
          imgSize &&
          objects &&
          objects.length > 0 &&
          containerRef.current && (
            <BoundingBoxOverlay
              objects={objects}
              naturalW={imgSize.w}
              naturalH={imgSize.h}
            />
          )}
      </div>

      {objects && objects.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {objects.map((obj, i) => (
            <span
              key={obj.id}
              className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px]"
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{
                  backgroundColor: BOX_COLORS[i % BOX_COLORS.length],
                }}
              />
              <span className="font-mono uppercase text-gray-300">
                {obj.object_type.replace(/_/g, " ")}
              </span>
              <span className="text-gray-500">
                {(obj.confidence * 100).toFixed(0)}%
              </span>
            </span>
          ))}
        </div>
      )}

      <div className="mt-2 space-y-1 text-xs text-gray-400">
        <MetaLine label="ID" value={metadata.image_id} />
        <MetaLine
          label="Dimensions"
          value={dimensionsLabel || "unknown"}
        />
        <MetaLine label="Format" value={metadata.format || "unknown"} />
        <MetaLine label="Source" value={metadata.source || "unknown"} />
      </div>
    </div>
  );
}

function BoundingBoxOverlay({
  objects,
  naturalW,
  naturalH,
}: {
  objects: DetectedObject[];
  naturalW: number;
  naturalH: number;
}) {
  return (
    <svg
      viewBox={`0 0 ${naturalW} ${naturalH}`}
      className="absolute inset-0 h-full w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      {objects.map((obj, i) => {
        const { x, y, width, height } = obj.geometry.box;
        const color = BOX_COLORS[i % BOX_COLORS.length];
        return (
          <g key={obj.id}>
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill="none"
              stroke={color}
              strokeWidth={Math.max(naturalW, naturalH) * 0.003}
              strokeDasharray={
                obj.object_type.includes("unknown") ? "8 4" : undefined
              }
            />
            <rect
              x={x}
              y={y - Math.max(naturalH * 0.035, 18)}
              width={width}
              height={Math.max(naturalH * 0.035, 18)}
              fill={color}
              rx={2}
            />
            <text
              x={x + 4}
              y={y - Math.max(naturalH * 0.035, 18) * 0.3}
              fill="#000"
              fontSize={Math.max(naturalH * 0.025, 12)}
              fontWeight="bold"
              fontFamily="monospace"
            >
              {obj.object_type.replace(/_/g, " ").toUpperCase()}{" "}
              {(obj.confidence * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function MetaLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono text-gray-300">{value}</span>
    </div>
  );
}

function SvgPlaceholder() {
  return (
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      className="text-gray-700"
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <path d="M21 15l-5-5L5 21" />
    </svg>
  );
}
