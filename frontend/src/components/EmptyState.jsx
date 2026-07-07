const SUGGESTIONS = [
  "Batch normalization vs layer normalization",
  "Explain the bias-variance tradeoff",
  "What are attention mechanisms and why do they matter?",
];

export default function EmptyState({ onSuggestion }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 px-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-from to-accent-to text-2xl font-bold shadow-lg shadow-indigo-950/40">
        C
      </div>
      <div>
        <h2 className="text-xl font-semibold text-slate-100">Clarity CRAG</h2>
        <p className="mt-1 max-w-sm text-sm text-slate-500">
          Corrective RAG that grades its own retrieval, falls back to the web when unsure, and only
          ever answers from evidence it trusts.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSuggestion(s)}
            className="rounded-full border border-border-subtle bg-surface-raised px-4 py-2 text-sm text-slate-300 transition hover:border-accent-from/50 hover:text-white"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
