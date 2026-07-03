import { useEffect, useState } from "react";

export default function Header() {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="flex items-center justify-between border-b border-dss-border bg-dss-panel px-6 py-3">
      <div className="flex items-center gap-3">
        <span className="text-2xl">&#x1F6E1;</span>
        <div>
          <h1 className="text-lg font-bold tracking-wide text-dss-accent">
            DSSPrototype
          </h1>
          <p className="text-[11px] text-dss-muted tracking-wider uppercase">
            HQ Mission Dashboard
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4 text-sm">
        <span className="rounded bg-dss-card px-2.5 py-1 font-mono text-xs text-dss-muted">
          v0.1.0
        </span>
        <time className="font-mono text-xs text-dss-muted" suppressHydrationWarning>
          {now.toISOString().replace("T", " ").slice(0, 19)}Z
        </time>
      </div>
    </header>
  );
}
