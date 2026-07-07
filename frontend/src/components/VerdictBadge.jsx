const STYLES = {
  CORRECT: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30",
  AMBIGUOUS: "bg-amber-500/15 text-amber-400 ring-amber-500/30",
  INCORRECT: "bg-rose-500/15 text-rose-400 ring-rose-500/30",
};

const LABELS = {
  CORRECT: "✅ Correct retrieval",
  AMBIGUOUS: "🤔 Ambiguous",
  INCORRECT: "🚫 Incorrect retrieval",
};

export default function VerdictBadge({ verdict, usedWebSearch }) {
  if (!verdict) return null;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      <span
        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STYLES[verdict] || STYLES.AMBIGUOUS}`}
      >
        {LABELS[verdict] || verdict}
      </span>
      {usedWebSearch && (
        <span className="inline-flex items-center rounded-full bg-sky-500/15 px-2.5 py-0.5 text-xs font-medium text-sky-400 ring-1 ring-inset ring-sky-500/30">
          🌐 Web search used
        </span>
      )}
    </div>
  );
}
