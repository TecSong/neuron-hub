const { useEffect, useMemo, useRef, useState } = React;

const WS_CANDIDATES = (() => {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const hostname = window.location.hostname || "localhost";
  const port = window.location.port ? `:${window.location.port}` : "";
  const sameOrigin = `${protocol}://${hostname}${port}/ws/chat`;
  const legacy = `${protocol}://${hostname}:8000/ws/chat`;
  const wsParam = new URLSearchParams(window.location.search).get("ws");
  const candidates = [wsParam, window.PKBA_WS_URL, sameOrigin, legacy].filter(Boolean);
  return Array.from(new Set(candidates));
})();

const EMPTY_FORM = {
  name: "",
  description: "",
  kb_dir: "",
  ollama_embed_model: "bge-m3",
  rerank_model: "BAAI/bge-reranker-v2-m3",
  chunk_size_tokens: "300",
  chunk_overlap_ratio: "0.15",
  vector_top_n: "8",
  bm25_top_n: "8",
  top_k: "4",
  dedup_similarity_threshold: "0.85",
  vector_weight: "0.6",
  bm25_weight: "0.4",
};

const toInt = (value, fallback) => {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toFloat = (value, fallback) => {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toForm = (config) => ({
  name: config?.name || "",
  description: config?.description || "",
  kb_dir: config?.kb_dir || "",
  ollama_embed_model: config?.ollama_embed_model || "bge-m3",
  rerank_model: config?.rerank_model || "BAAI/bge-reranker-v2-m3",
  chunk_size_tokens: String(config?.chunk_size_tokens ?? "300"),
  chunk_overlap_ratio: String(config?.chunk_overlap_ratio ?? "0.15"),
  vector_top_n: String(config?.vector_top_n ?? "8"),
  bm25_top_n: String(config?.bm25_top_n ?? "8"),
  top_k: String(config?.top_k ?? "4"),
  dedup_similarity_threshold: String(config?.dedup_similarity_threshold ?? "0.85"),
  vector_weight: String(config?.vector_weight ?? "0.6"),
  bm25_weight: String(config?.bm25_weight ?? "0.4"),
});

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

function App() {
  const [view, setView] = useState("configs");
  const [configs, setConfigs] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [activateOnCreate, setActivateOnCreate] = useState(true);

  const activeConfig = useMemo(
    () => configs.find((config) => config.id === activeId) || null,
    [configs, activeId]
  );

  const loadConfigs = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await apiFetch("/api/configs");
      setConfigs(data.configs || []);
      setActiveId(data.active_id || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfigs();
  }, []);

  const resetForm = (config = null) => {
    setEditingId(config?.id || null);
    setForm(config ? toForm(config) : { ...EMPTY_FORM });
  };

  const handleInput = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const buildPayload = () => ({
    name: form.name.trim(),
    description: form.description.trim(),
    kb_dir: form.kb_dir.trim(),
    ollama_embed_model: form.ollama_embed_model.trim(),
    rerank_model: form.rerank_model.trim(),
    chunk_size_tokens: toInt(form.chunk_size_tokens, 300),
    chunk_overlap_ratio: toFloat(form.chunk_overlap_ratio, 0.15),
    vector_top_n: toInt(form.vector_top_n, 8),
    bm25_top_n: toInt(form.bm25_top_n, 8),
    top_k: toInt(form.top_k, 4),
    dedup_similarity_threshold: toFloat(form.dedup_similarity_threshold, 0.85),
    vector_weight: toFloat(form.vector_weight, 0.6),
    bm25_weight: toFloat(form.bm25_weight, 0.4),
  });

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus("");
    setError("");
    try {
      if (editingId) {
        await apiFetch(`/api/configs/${editingId}`, {
          method: "PUT",
          body: JSON.stringify(buildPayload()),
        });
        setStatus("Config updated.");
      } else {
        await apiFetch("/api/configs", {
          method: "POST",
          body: JSON.stringify({ ...buildPayload(), activate: activateOnCreate }),
        });
        setStatus("Config created.");
      }
      resetForm(null);
      await loadConfigs();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleActivate = async (configId) => {
    setError("");
    try {
      await apiFetch(`/api/configs/${configId}/activate`, { method: "POST" });
      await loadConfigs();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDelete = async (configId) => {
    if (!window.confirm("Delete this config?")) return;
    setError("");
    try {
      await apiFetch(`/api/configs/${configId}`, { method: "DELETE" });
      await loadConfigs();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleIngest = async () => {
    setStatus("Running ingestion...");
    setError("");
    try {
      const data = await apiFetch("/api/ingest", { method: "POST" });
      setStatus(`Indexed ${data.file_count} files into ${data.chunk_count} chunks.`);
    } catch (err) {
      setError(err.message);
      setStatus("");
    }
  };

  return (
    <div className="app-shell">
      <div className="hero">
        <div>
          <h1>PKBA Console</h1>
          <p>
            Manage personal knowledge base profiles and chat with your active KB
            over WebSocket streaming.
          </p>
        </div>
        <div className="nav-tabs">
          <button
            className={view === "configs" ? "active" : ""}
            onClick={() => setView("configs")}
          >
            Configs
          </button>
          <button
            className={view === "chat" ? "active" : ""}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
        </div>
      </div>

      {error && <div className="panel">{error}</div>}
      {status && <div className="panel">{status}</div>}

      {view === "configs" && (
        <div className="content">
          <div className="panel">
            <h2>Knowledge Base Profiles</h2>
            <p className="status">
              {loading ? "Loading configs..." : "Manage and activate KB profiles."}
            </p>
            <div className="actions" style={{ marginBottom: "16px" }}>
              <button className="primary" onClick={() => resetForm(null)}>
                New Config
              </button>
              <button className="secondary" onClick={handleIngest} disabled={!activeId}>
                Ingest Active KB
              </button>
            </div>
            <div className="grid">
              {configs.map((config) => (
                <div
                  key={config.id}
                  className={`card ${config.id === activeId ? "active" : ""}`}
                >
                  <h3>{config.name || "Untitled"}</h3>
                  <div className="meta">ID: {config.id}</div>
                  <div className="meta">Path: {config.kb_dir || "Not set"}</div>
                  <div className="meta">
                    Embed: {config.ollama_embed_model} / Rerank: {config.rerank_model}
                  </div>
                  <div className="actions">
                    {config.id !== activeId ? (
                      <button className="primary" onClick={() => handleActivate(config.id)}>
                        Activate
                      </button>
                    ) : (
                      <button className="secondary" disabled>
                        Active
                      </button>
                    )}
                    <button className="secondary" onClick={() => resetForm(config)}>
                      Edit
                    </button>
                    <button className="warn" onClick={() => handleDelete(config.id)}>
                      Delete
                    </button>
                  </div>
                </div>
              ))}
              {!configs.length && <div className="card">No configs found.</div>}
            </div>
          </div>

          <div className="panel">
            <h2>{editingId ? "Edit Config" : "Create Config"}</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div>
                  <label>Name</label>
                  <input name="name" value={form.name} onChange={handleInput} required />
                </div>
                <div>
                  <label>KB Directory</label>
                  <input name="kb_dir" value={form.kb_dir} onChange={handleInput} />
                </div>
                <div>
                  <label>Embedding Model</label>
                  <input
                    name="ollama_embed_model"
                    value={form.ollama_embed_model}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>Rerank Model</label>
                  <input
                    name="rerank_model"
                    value={form.rerank_model}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>Chunk Size Tokens</label>
                  <input
                    name="chunk_size_tokens"
                    value={form.chunk_size_tokens}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>Chunk Overlap Ratio</label>
                  <input
                    name="chunk_overlap_ratio"
                    value={form.chunk_overlap_ratio}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>Vector Top N</label>
                  <input name="vector_top_n" value={form.vector_top_n} onChange={handleInput} />
                </div>
                <div>
                  <label>BM25 Top N</label>
                  <input name="bm25_top_n" value={form.bm25_top_n} onChange={handleInput} />
                </div>
                <div>
                  <label>Top K</label>
                  <input name="top_k" value={form.top_k} onChange={handleInput} />
                </div>
                <div>
                  <label>Dedup Similarity</label>
                  <input
                    name="dedup_similarity_threshold"
                    value={form.dedup_similarity_threshold}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>Vector Weight</label>
                  <input
                    name="vector_weight"
                    value={form.vector_weight}
                    onChange={handleInput}
                  />
                </div>
                <div>
                  <label>BM25 Weight</label>
                  <input name="bm25_weight" value={form.bm25_weight} onChange={handleInput} />
                </div>
              </div>
              <div>
                <label>Description</label>
                <textarea name="description" value={form.description} onChange={handleInput} />
              </div>
              {!editingId && (
                <div>
                  <label>
                    <input
                      type="checkbox"
                      checked={activateOnCreate}
                      onChange={(event) => setActivateOnCreate(event.target.checked)}
                    />{" "}
                    Activate after create
                  </label>
                </div>
              )}
              <div className="actions">
                <button className="primary" type="submit">
                  {editingId ? "Save Changes" : "Create Config"}
                </button>
                <button className="secondary" type="button" onClick={() => resetForm(null)}>
                  Clear
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {view === "chat" && (
        <ChatPanel activeConfig={activeConfig} />
      )}
    </div>
  );
}

function ChatPanel({ activeConfig }) {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(true);
  const [wsUrl, setWsUrl] = useState(WS_CANDIDATES[0] || "");
  const [messages, setMessages] = useState([]);
  const [turns, setTurns] = useState([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const wsRef = useRef(null);
  const pendingQuestionRef = useRef(null);

  useEffect(() => {
    let active = true;
    let ws = null;
    let index = 0;
    let opened = false;

    const connect = () => {
      if (!active) return;
      if (index >= WS_CANDIDATES.length) {
        setConnecting(false);
        setConnected(false);
        setError("WebSocket connection failed. Check WS URL or server status.");
        return;
      }
      const url = WS_CANDIDATES[index];
      setWsUrl(url);
      setConnecting(true);
      opened = false;
      ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        if (!active) return;
        opened = true;
        setConnecting(false);
        setConnected(true);
        setError("");
      };
      ws.onclose = (event) => {
        if (!active) return;
        setConnected(false);
        if (!opened) {
          index += 1;
          connect();
          return;
        }
        setConnecting(false);
        if (event.code && event.code !== 1000) {
          setError(`WebSocket closed (code ${event.code}).`);
        }
      };
      ws.onerror = () => {
        if (!active) return;
        setConnected(false);
        if (!opened) {
          return;
        }
        setError("WebSocket error.");
      };
      ws.onmessage = (event) => {
        if (!active) return;
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const lastIndex = next.length - 1;
              if (lastIndex >= 0 && next[lastIndex].role === "assistant") {
                next[lastIndex] = {
                  ...next[lastIndex],
                  content: `${next[lastIndex].content}${payload.content}`,
                };
              }
              return next;
            });
          }
          if (payload.type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              const lastIndex = next.length - 1;
              if (lastIndex >= 0 && next[lastIndex].role === "assistant") {
                next[lastIndex] = {
                  ...next[lastIndex],
                  content: payload.answer,
                  sources: payload.sources || [],
                };
              }
              return next;
            });
            if (pendingQuestionRef.current) {
              setTurns((prev) => [
                ...prev,
                { question: pendingQuestionRef.current, answer: payload.answer },
              ]);
              pendingQuestionRef.current = null;
            }
          }
          if (payload.type === "error") {
            setError(payload.message || "Server error.");
          }
        } catch (err) {
          setError("Invalid server response.");
        }
      };
    };

    connect();

    return () => {
      active = false;
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const sendMessage = () => {
    const question = input.trim();
    if (!question) {
      return;
    }
    if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError("WebSocket not connected.");
      return;
    }
    setError("");
    pendingQuestionRef.current = question;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    wsRef.current.send(
      JSON.stringify({ question, history: turns, return_sources: true })
    );
    setInput("");
  };

  return (
    <div className="panel">
      <h2>Chat Workspace</h2>
      <p className="status">
        {connected ? "Connected" : connecting ? "Connecting" : "Disconnected"} | Active KB:{" "}
        {activeConfig?.name || "None"} | WS: {wsUrl || "Not set"}
      </p>
      {error && <div className="status">{error}</div>}
      <div className="chat-window">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
            {message.content || "..."}
            {message.sources && message.sources.length > 0 && (
              <small>
                Sources:{" "}
                {message.sources
                  .slice(0, 3)
                  .map((source) => source.source)
                  .join(" | ")}
              </small>
            )}
          </div>
        ))}
        {!messages.length && (
          <div className="message assistant">Ask a question to start.</div>
        )}
      </div>
      <div className="chat-input">
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask something from your knowledge base..."
          onKeyDown={(event) => {
            if (event.key === "Enter") sendMessage();
          }}
        />
        <button className="primary" onClick={sendMessage} disabled={!connected}>
          Send
        </button>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
