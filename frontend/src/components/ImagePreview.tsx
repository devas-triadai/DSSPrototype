import type { ImageMetadata } from "../types";

interface ImagePreviewProps {
  metadata: ImageMetadata;
}

export default function ImagePreview({ metadata }: ImagePreviewProps) {
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

      <div className="flex aspect-video items-center justify-center rounded border border-dss-border bg-gray-900/50">
        <div className="text-center">
          <div className="text-3xl text-gray-700">
            <SvgPlaceholder />
          </div>
          <p className="mt-1 text-[11px] text-gray-600 font-mono">
            {metadata.image_id}
          </p>
        </div>
      </div>

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
