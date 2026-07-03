import { useCallback, useEffect, useRef, useState } from "react";
import type { ImageMetadata } from "../types";
import LoadingSpinner from "./LoadingSpinner";

interface ImageUploadPanelProps {
  onExecute: (file: File, metadata: ImageMetadata) => void;
  isRunning: boolean;
}

interface FileMeta {
  file: File;
  previewUrl: string;
  metadata: ImageMetadata;
}

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

let idCounter = 0;

function generateImageId(): string {
  idCounter += 1;
  return `upload_${Date.now()}_${idCounter}`;
}

function readFileMetadata(file: File): Promise<FileMeta> {
  return new Promise((resolve, reject) => {
    const previewUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const ext = file.type.replace("image/", "").toUpperCase();
      resolve({
        file,
        previewUrl,
        metadata: {
          image_id: generateImageId(),
          timestamp: new Date().toISOString(),
          source: "local_upload",
          width: img.naturalWidth,
          height: img.naturalHeight,
          format: ext === "JPEG" ? "JPEG" : ext,
        },
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(previewUrl);
      reject(new Error("Failed to decode image"));
    };
    img.src = previewUrl;
  });
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

export default function ImageUploadPanel({
  onExecute,
  isRunning,
}: ImageUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileMeta, setFileMeta] = useState<FileMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    return () => {
      if (fileMeta) URL.revokeObjectURL(fileMeta.previewUrl);
    };
  }, [fileMeta]);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(
        `Unsupported format "${file.type}". Accepted: JPEG, PNG, BMP, WebP`,
      );
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setError("File exceeds 50 MB limit");
      return;
    }
    setIsProcessing(true);
    try {
      if (fileMeta) URL.revokeObjectURL(fileMeta.previewUrl);
      const meta = await readFileMetadata(file);
      setFileMeta(meta);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to read image");
    } finally {
      setIsProcessing(false);
    }
  }, [fileMeta]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleExecute = useCallback(() => {
    if (fileMeta && !isRunning) {
      onExecute(fileMeta.file, fileMeta.metadata);
    }
  }, [fileMeta, isRunning, onExecute]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const canExecute = fileMeta !== null && !isRunning && !isProcessing;

  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Upload Image
      </h3>

      {/* Drop zone + file input */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="flex flex-col items-center justify-center rounded border border-dashed border-gray-700 bg-gray-900/50 px-4 py-6"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.bmp,.webp"
          onChange={handleChange}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isRunning}
          className="rounded bg-dss-accent px-4 py-2 text-xs font-bold text-gray-950 transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Choose Image
        </button>
        <p className="mt-2 text-[10px] text-gray-600">
          or drag & drop — JPEG, PNG, BMP, WebP (max 50 MB)
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-3 rounded bg-red-950/30 px-3 py-2 text-xs text-dss-danger">
          {error}
        </div>
      )}

      {/* Processing */}
      {isProcessing && (
        <div className="mt-3">
          <LoadingSpinner size="sm" text="Reading image\u2026" />
        </div>
      )}

      {/* Preview + metadata */}
      {fileMeta && !isProcessing && (
        <>
          <div className="mt-3 overflow-hidden rounded border border-dss-border">
            <img
              src={fileMeta.previewUrl}
              alt={fileMeta.file.name}
              className="aspect-video w-full object-cover"
            />
          </div>

          <div className="mt-2 space-y-1 text-xs text-gray-400">
            <MetaLine label="Filename" value={fileMeta.file.name} />
            <MetaLine
              label="Dimensions"
              value={`${fileMeta.metadata.width} x ${fileMeta.metadata.height}`}
            />
            <MetaLine
              label="Format"
              value={fileMeta.metadata.format ?? "unknown"}
            />
            <MetaLine
              label="File Size"
              value={formatFileSize(fileMeta.file.size)}
            />
            <MetaLine
              label="Image ID"
              value={fileMeta.metadata.image_id}
            />
            <MetaLine
              label="Timestamp"
              value={fileMeta.metadata.timestamp}
            />
            <MetaLine label="Source" value="Local Upload" />
          </div>

          <button
            type="button"
            onClick={handleExecute}
            disabled={!canExecute}
            className="mt-4 w-full rounded bg-dss-accent px-3 py-2 text-xs font-bold text-gray-950 transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRunning ? "Executing Pipeline\u2026" : "Execute Pipeline"}
          </button>
        </>
      )}

      {isRunning && <LoadingSpinner size="sm" text="Executing pipeline\u2026" />}
    </div>
  );
}

function MetaLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="truncate font-mono text-gray-300 max-w-[60%]">
        {value}
      </span>
    </div>
  );
}
