const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const listThreads = () => request("/threads");

export const createThread = () =>
  request("/threads", { method: "POST" });

export const getThread = (threadId) => request(`/threads/${threadId}`);

export const sendMessage = (threadId, content) =>
  request(`/threads/${threadId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
