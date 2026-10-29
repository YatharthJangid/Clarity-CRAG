import re
from typing import List

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from pydantic import BaseModel

from app.config import settings
from app.crag.llm import llm, structured
from app.crag.prompts import answer_prompt, doc_eval_prompt, filter_prompt, rewrite_prompt
from app.crag.state import State
from app.crag.vectorstore import get_retriever

# -----------------------------
# Retrieve
# -----------------------------
def retrieve_node(state: State) -> dict:
    return {"docs": get_retriever().invoke(state["question"])}


# -----------------------------
# Score-based doc evaluator
# -----------------------------
class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_chain = doc_eval_prompt | structured(DocEvalScore)


def eval_each_doc_node(state: State) -> dict:
    q = state["question"]
    docs = state["docs"]
    scores: List[float] = []
    good: List[Document] = []

    if not docs:
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No documents retrieved from vector store.",
        }

    inputs = [{"question": q, "chunk": d.page_content} for d in docs]
    try:
        results = doc_eval_chain.batch(inputs, config={"max_concurrency": 4}, return_exceptions=True)
    except Exception:
        results = [None] * len(docs)

    for d, out in zip(docs, results):
        score = out.score if (out and not isinstance(out, Exception) and hasattr(out, "score")) else 0.0
        scores.append(score)
        if score > settings.lower_threshold:
            good.append(d)

    if any(s > settings.upper_threshold for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {settings.upper_threshold}.",
        }

    if len(scores) > 0 and all(s < settings.lower_threshold for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {settings.lower_threshold}.",
        }

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {settings.upper_threshold}, but not all were < {settings.lower_threshold}.",
    }


# -----------------------------
# Sentence-level decomposer + LLM filter
# -----------------------------
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class RefinedStrips(BaseModel):
    kept_sentences: List[str]


filter_chain = filter_prompt | structured(RefinedStrips)


# -----------------------------
# Knowledge refinement
# CORRECT   => internal docs only
# INCORRECT => web docs only
# AMBIGUOUS => internal + web docs
# -----------------------------
def refine(state: State) -> dict:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)[:20]

    kept: List[str] = []
    if strips:
        formatted_sentences = "\n".join(f"- {s}" for s in strips)
        try:
            out = filter_chain.invoke({"question": q, "sentences": formatted_sentences})
            if out and hasattr(out, "kept_sentences") and out.kept_sentences:
                kept = out.kept_sentences
            else:
                kept = strips
        except Exception:
            kept = strips
    else:
        kept = []

    refined_context = "\n".join(kept).strip()

    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


# -----------------------------
# Query rewrite for web search
# -----------------------------
class WebQuery(BaseModel):
    query: str


rewrite_chain = rewrite_prompt | structured(WebQuery)


def rewrite_query_node(state: State) -> dict:
    try:
        out = rewrite_chain.invoke({"question": state["question"]})
        if out and hasattr(out, "query") and out.query:
            return {"web_query": out.query}
    except Exception:
        pass
    return {"web_query": state["question"]}


# -----------------------------
# Web search
# -----------------------------
tavily = TavilySearch(max_results=5, tavily_api_key=settings.tavily_api_key)


def web_search_node(state: State) -> dict:
    q = state.get("web_query") or state["question"]
    raw = tavily.invoke({"query": q})
    results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])

    web_docs: List[Document] = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}


# -----------------------------
# Generate
# -----------------------------
def generate(state: State) -> dict:
    out = (answer_prompt | llm).invoke({"question": state["question"], "context": state["refined_context"]})
    return {"answer": out.content}


# -----------------------------
# Record turn into per-thread history (accumulates via the State's
# operator.add reducer + the SqliteSaver checkpointer)
# -----------------------------
def record_turn(state: State) -> dict:
    return {
        "history": [
            {"role": "user", "content": state["question"]},
            {
                "role": "assistant",
                "content": state["answer"],
                "verdict": state["verdict"],
                "used_web_search": state["verdict"] != "CORRECT",
            },
        ]
    }


# -----------------------------
# Routing
# CORRECT            => refine
# INCORRECT/AMBIGUOUS => rewrite_query -> web_search -> refine
# -----------------------------
def route_after_eval(state: State) -> str:
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"
