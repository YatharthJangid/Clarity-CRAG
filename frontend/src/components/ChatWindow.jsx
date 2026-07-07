import { useEffect, useRef, useState } from "react";
import MessageBubble from "./MessageBubble";

function ThinkingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-border-subtle bg-surface-raised px-4 py-3">
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-slate-400 [animation-delay:0ms]" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-slate-400 [animation-delay:150ms]" />
        <span className="thinking-dot h-1.5 w-1.5 rounded-full bg-slate-400 [animation-delay:300ms]" />
      </div>
    </div>
  );
}

export default function ChatWindow({ title, messages, loading, onSend }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    onSend(text);
  };

  return (
    <div className="flex h-full flex-1 flex-col">
      <header className="flex items-center border-b border-border-subtle px-6 py-4">
        <h2 className="truncate text-sm font-medium text-slate-200">{title}</h2>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-5">
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          {loading && <ThinkingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      <form onSubmit={submit} className="border-t border-border-subtle px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-2xl border border-border-subtle bg-surface-raised px-4 py-2 focus-within:border-accent-from/50">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(e);
              }
            }}
            rows={1}
            placeholder="Ask about your documents..."
            className="max-h-40 flex-1 resize-none bg-transparent py-2 text-sm text-slate-100 placeholder-slate-500 outline-none"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="mb-1 rounded-lg bg-gradient-to-r from-accent-from to-accent-to px-4 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
