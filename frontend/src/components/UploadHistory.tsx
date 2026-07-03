interface UploadEntry {
  imageId: string;
  filename: string;
  timestamp: string;
  status: string;
}

interface UploadHistoryProps {
  entries: UploadEntry[];
  onSelect: (imageId: string) => void;
}

export default function UploadHistory({
  entries,
  onSelect,
}: UploadHistoryProps) {
  if (entries.length === 0) {
    return (
      <div className="rounded-lg border border-dss-border bg-dss-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
          Upload History
        </h3>
        <p className="text-xs text-gray-600">No executions yet</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Upload History
      </h3>

      <div className="space-y-1">
        {entries.map((entry, idx) => (
          <button
            key={`${entry.imageId}-${idx}`}
            type="button"
            onClick={() => onSelect(entry.imageId)}
            className="flex w-full items-center justify-between rounded px-2 py-1.5 text-xs transition-colors hover:bg-gray-800"
          >
            <span className="truncate font-mono text-gray-300 max-w-[55%]">
              {entry.filename}
            </span>
            <div className="flex items-center gap-2 shrink-0">
              <span
                className={`text-[10px] uppercase ${
                  entry.status === "completed"
                    ? "text-dss-success"
                    : "text-dss-danger"
                }`}
              >
                {entry.status}
              </span>
              <span className="text-[10px] text-gray-600">
                {entry.timestamp.slice(11, 19)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
