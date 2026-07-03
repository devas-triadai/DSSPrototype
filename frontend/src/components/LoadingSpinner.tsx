interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  text?: string;
}

const sizeMap = {
  sm: "h-4 w-4 border-2",
  md: "h-8 w-8 border-3",
  lg: "h-12 w-12 border-4",
};

export default function LoadingSpinner({
  size = "md",
  text,
}: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-8">
      <div
        className={`animate-spin rounded-full border-dss-border border-t-dss-accent ${sizeMap[size]}`}
      />
      {text && <p className="text-sm text-dss-muted">{text}</p>}
    </div>
  );
}
