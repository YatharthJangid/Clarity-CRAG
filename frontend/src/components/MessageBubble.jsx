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
          <p className="whitespace-pre-wrap text-[0.95rem] leading-relaxed">{message.content}</p>
        </div>
        {!isUser && <VerdictBadge verdict={message.verdict} usedWebSearch={message.used_web_search} />}
      </div>
    </div>
  );
}
