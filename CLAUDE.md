# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Clarity CRAG is a Corrective RAG (CRAG) chat app: a FastAPI + LangGraph backend that grades its
own retrieval, falls back to live web search when retrieval isn't trustworthy, and only answers
from evidence it has actually filtered for relevance. It implements the CRAG paper
(arXiv:2401.15884) end-to-end, backed by a React/Vite chat UI. See `README.md` for the full
picture (paper background, architecture diagram, API reference, troubleshooting) — this file
only covers what's needed to work on the code productively.

## Commands

### Backend
```bash
cd backend
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in HF_TOKEN, GEMINI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY

python scripts/ingest.py            # one-time: embed documents/*.pdf into Pinecone
python scripts/save_graph_image.py  # regenerate assets/graph.png (no API keys needed)
uvicorn app.main:app --reload --port 8000
```
No test suite exists for the backend currently. There's no lint config for it either.

### Frontend
```bash
cd frontend
npm install
npm run dev      # Vite dev server on 5173, proxies /api -> localhost:8000
npm run build
npm run lint      # oxlint, zero-config
```

### Evaluation (Naive RAG vs CRAG, scored with ragas)
```bash
source .venv/bin/activate
python evaluation/run_eval.py   # needs backend/.env filled in + a populated Pinecone index
```
Writes `evaluation/results/{comparison_table.md,.csv, per_question_*.csv, metrics_chart.png}`.

## Architecture

### The CRAG pipeline is a LangGraph state machine
`backend/app/crag/{state,prompts,nodes,graph}.py` together define one `StateGraph` — read all
four before changing pipeline behavior, since routing logic lives in `graph.py` but the verdict
it routes on is computed in `nodes.py`, against thresholds from `config.py`.

Flow: `retrieve` → `eval_each_doc` (LLM-grades every retrieved chunk 0.0-1.0) → conditional
routing on the verdict:
- **CORRECT** (any chunk scored > `UPPER_TH`) → straight to `refine`
- **INCORRECT** (all chunks scored < `LOWER_TH`) / **AMBIGUOUS** (mixed) → `rewrite_query` →
  `web_search` (Tavily) → `refine`

`refine` decomposes the surviving context into sentence "strips" and LLM-filters each one
independently before `generate` answers strictly from the kept strips. `record_turn` then
appends `{role, content, verdict, used_web_search}` to `state["history"]`, which is an
`Annotated[List[dict], operator.add]` reducer — LangGraph accumulates it turn-over-turn
automatically as long as invocations share a `thread_id`.

**Short-term memory** is the `SqliteSaver` checkpointer in `graph.py` (`build_graph()`,
`@lru_cache`d singleton) persisting the entire graph state per `thread_id` to
`backend/data/checkpoints.db`. This is separate from `app/db.py`, a small hand-rolled sqlite
store (`backend/data/threads.db`) that only tracks thread *metadata* (title/timestamps) for the
sidebar — chat content itself lives exclusively in the checkpointer, not in `db.py`.

### LLM chain: five fallback tiers, deliberately in this order
`backend/app/crag/llm.py` chains chat models with `.with_fallbacks()`:

1. **HuggingFace loop** (`HF_CHAT_MODELS`, comma-separated, default 3 open models routed via
   HF Inference Providers) — primary, chosen for free high-volume throughput.
2. **Gemini** (`CHAT_MODEL`, default `gemini-2.5-flash`) — `max_retries=1` deliberately, so a
   429 fails fast into the next tier instead of burning ~20s/call on Google's own backoff.
   Gemini's free tier is very restrictive (~20 req/day), easily exhausted since one CRAG
   question makes 10-30+ LLM calls (grade every doc + filter every strip).
3. **Groq** (`FALLBACK_CHAT_MODEL`, default `qwen/qwen3-32b`) — last tier.

`structured(schema)` builds the equivalent fallback chain for structured/tool-calling output
(used by every node except `generate`/`web_search`). Two non-obvious things baked in here:
- HuggingFace's `with_structured_output` doesn't accept a Pydantic class for tool-calling
  (raises `NotImplementedError`) — `_hf_structured()` converts the schema to a dict, calls with
  `method="function_calling"`, then coerces the dict result back into the Pydantic class via a
  `RunnableLambda`, so callers get the same type regardless of which tier answered.
- Open-weight models occasionally mis-type a tool-call argument (e.g. emit the string `"false"`
  instead of boolean `false`), which the provider rejects server-side before any response
  reaches this code. Both the HF and Groq structured tiers wrap themselves in
  `.with_retry(stop_after_attempt=2)` to absorb this instead of failing the whole chain.

`primary_llm` (Gemini) and `fallback_llm` (Groq) are also exported directly — `run_eval.py`
uses `primary_llm` for ragas scoring specifically because `llm`/`structured()` are
`RunnableWithFallbacks`, not a `BaseChatModel`, and ragas expects the latter.

### Embeddings: dimension is a cross-cutting constraint
`backend/app/crag/vectorstore.py` wraps Gemini's `gemini-embedding-001` (primary) and FastEmbed's
`jinaai/jina-embeddings-v2-small-en` (local ONNX fallback, ~no dependencies) in a
`FallbackEmbeddings` class. **Both must produce vectors of exactly `EMBEDDING_DIMENSION`**
(default 512) since either one can serve any given request against the same Pinecone index —
Gemini's dimension is truncated via Matryoshka Representation Learning, passed as
`output_dimensionality` at call time (passing it to the constructor is silently ignored by the
SDK, a real gotcha). If you change `EMBEDDING_MODEL`, `FALLBACK_EMBEDDING_MODEL`, or
`EMBEDDING_DIMENSION`, the Pinecone index must be deleted and recreated, then
`scripts/ingest.py` re-run — a stale index silently mismatches dimensions at query time.

### Everything is env-driven through one Settings object
`backend/app/config.py` is the single source of truth for every tunable (models, thresholds,
`RETRIEVER_K`, CORS origins, file paths) and loads `backend/.env` via `python-dotenv`. There is
no other config mechanism — new tunables should be added here, not hardcoded in `nodes.py` or
elsewhere.

### `iterative_notebooks/` is historical, not source
The six notebooks (`1_basic_rag` → `6_ambiguous`) are the unmodified research trail the
production pipeline was ported from. `backend/app/crag/prompts.py` and the routing logic in
`nodes.py`/`graph.py` are a 1:1 port of notebook 6's logic — don't treat the notebooks as needing
to stay in sync with the backend; they're preserved as-is for reference only.
