import ReactMarkdown from "react-markdown";
import VerdictBadge from "./VerdictBadge";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex animate-fade-in-up ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[75%] ${isUser ? "order-2" : "order-1"}`}>
        <div
          className={
            isUser
              ? "rounded-2xl rounded-tr-sm bg-gradient-to-br from-accent-from to-accent-to px-4 py-2.5 text-white shadow-lg shadow-indigo-950/30"
              : "rounded-2xl rounded-tl-sm border border-border-subtle bg-surface-raised px-4 py-2.5 text-slate-100"
          }
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-[0.95rem] leading-relaxed">{message.content}</p>
          ) : (
            <div className="text-[0.95rem] leading-relaxed">
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                  ul: ({ children }) => <ul className="my-2 list-disc pl-5 space-y-1">{children}</ul>,
                  ol: ({ children }) => <ol className="my-2 list-decimal pl-5 space-y-1">{children}</ol>,
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                  code: ({ inline, children }) =>
                    inline ? (
                      <code className="rounded bg-black/40 px-1.5 py-0.5 font-mono text-xs text-indigo-300">
                        {children}
                      </code>
                    ) : (
                      <pre className="my-2 overflow-x-auto rounded-lg bg-black/50 p-3 font-mono text-xs text-slate-200">
                        <code>{children}</code>
                      </pre>
                    ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && <VerdictBadge verdict={message.verdict} usedWebSearch={message.used_web_search} />}
      </div>
    </div>
  );
}
