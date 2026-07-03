import type { DecisionRecommendation } from "../types";

interface DecisionRecommendationPanelProps {
  decision: DecisionRecommendation | null | undefined;
}

export default function DecisionRecommendationPanel({
  decision,
}: DecisionRecommendationPanelProps) {
  return (
    <div className="rounded-lg border border-dss-border bg-dss-card p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dss-accent">
        Decision Recommendation
      </h3>

      {!decision ? (
        <p className="text-xs text-gray-600">
          No decision recommendation available
        </p>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <span className="rounded bg-amber-950/30 px-2 py-0.5 font-mono text-[11px] text-amber-400">
              Priority {decision.priority}
            </span>
            <span className="text-[10px] text-gray-500 font-mono">
              {decision.recommendation_id.slice(0, 12)}
            </span>
          </div>

          <div>
            <p className="mb-1 text-[10px] text-gray-500">
              Recommended Actions
            </p>
            <ul className="space-y-1">
              {decision.recommended_actions.map((action, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 rounded bg-gray-800/50 px-2.5 py-1.5"
                >
                  <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-dss-accent/20 text-[10px] font-bold text-dss-accent">
                    {i + 1}
                  </span>
                  <span className="text-xs leading-relaxed text-gray-200">
                    {action}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className="mb-0.5 text-[10px] text-gray-500">Reasoning</p>
            <p className="text-xs leading-relaxed text-gray-300">
              {decision.reason}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
