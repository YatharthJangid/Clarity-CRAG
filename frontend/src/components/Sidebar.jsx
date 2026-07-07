function timeAgo(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export default function Sidebar({ threads, activeThreadId, onNewChat, onSelectThread }) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-border-subtle bg-surface-sidebar">
      <div className="flex items-center gap-2 px-5 pt-6 pb-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-from to-accent-to text-sm font-bold">
          C
        </div>
        <div>
          <h1 className="text-sm font-semibold text-slate-100">Clarity CRAG</h1>
          <p className="text-xs text-slate-500">Self-Correcting Study Assistant</p>
        </div>
      </div>

      <div className="px-3">
        <button
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent-from to-accent-to px-3 py-2.5 text-sm font-medium text-white shadow-md shadow-indigo-950/40 transition hover:opacity-90 active:scale-[0.98]"
        >
          <span className="text-base leading-none">+</span> New Chat
        </button>
      </div>

      <div className="mt-4 flex-1 overflow-y-auto px-3 pb-4">
        <p className="mb-2 px-2 text-xs font-medium tracking-wide text-slate-500 uppercase">Chats</p>
        <div className="flex flex-col gap-1">
          {threads.length === 0 && (
            <p className="px-2 py-4 text-center text-xs text-slate-600">No chats yet — start one above.</p>
          )}
          {threads.map((t) => (
            <button
              key={t.thread_id}
              onClick={() => onSelectThread(t.thread_id)}
              className={`group flex flex-col rounded-lg px-3 py-2 text-left transition ${
                t.thread_id === activeThreadId
                  ? "bg-surface-raised ring-1 ring-accent-from/40"
                  : "hover:bg-surface-raised/60"
              }`}
            >
              <span className="truncate text-sm text-slate-200">{t.title}</span>
              <span className="text-xs text-slate-500">{timeAgo(t.updated_at)}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
