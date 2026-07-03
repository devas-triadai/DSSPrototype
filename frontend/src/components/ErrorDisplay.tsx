interface ErrorDisplayProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorDisplay({ message, onRetry }: ErrorDisplayProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dss-danger/30 bg-red-950/20 p-6">
      <div className="flex items-center gap-2">
        <span className="text-lg text-dss-danger">&#x2716;</span>
        <p className="text-sm font-medium text-red-400">{message}</p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded bg-red-900/40 px-4 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-800/50"
        >
          Retry
        </button>
      )}
    </div>
  );
}
