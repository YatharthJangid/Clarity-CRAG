import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import EmptyState from "./components/EmptyState";
import { createThread, getThread, listThreads, sendMessage } from "./lib/api";

export default function App() {
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [title, setTitle] = useState("New Chat");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refreshThreads = () => listThreads().then(setThreads).catch((e) => setError(e.message));

  useEffect(() => {
    refreshThreads();
  }, []);

  const handleNewChat = async () => {
    try {
      const thread = await createThread();
      setThreads((prev) => [thread, ...prev]);
      setActiveThreadId(thread.thread_id);
      setTitle(thread.title);
      setMessages([]);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleSelectThread = async (threadId) => {
    try {
      const detail = await getThread(threadId);
      setActiveThreadId(threadId);
      setTitle(detail.title);
      setMessages(detail.messages);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleSend = async (text) => {
    setError(null);
    let threadId = activeThreadId;

    if (!threadId) {
      const thread = await createThread();
      threadId = thread.thread_id;
      setThreads((prev) => [thread, ...prev]);
      setActiveThreadId(threadId);
    }

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await sendMessage(threadId, text);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          verdict: res.verdict,
          used_web_search: res.used_web_search,
        },
      ]);
      refreshThreads();
      if (title === "New Chat") setTitle(text.slice(0, 50));
    } catch (e) {
      setError(e.message);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Something went wrong: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-surface">
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        onNewChat={handleNewChat}
        onSelectThread={handleSelectThread}
      />

      <main className="flex flex-1 flex-col">
        {error && (
          <div className="border-b border-rose-900/40 bg-rose-950/40 px-6 py-2 text-center text-xs text-rose-300">
            {error}
          </div>
        )}
        {activeThreadId ? (
          <ChatWindow title={title} messages={messages} loading={loading} onSend={handleSend} />
        ) : (
          <EmptyState onSuggestion={handleSend} />
        )}
      </main>
    </div>
  );
}
