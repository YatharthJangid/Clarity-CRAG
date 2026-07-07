import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.crag.nodes import (
    eval_each_doc_node,
    generate,
    record_turn,
    refine,
    retrieve_node,
    rewrite_query_node,
    route_after_eval,
    web_search_node,
)
from app.crag.state import State


def _build_uncompiled_graph() -> StateGraph:
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)
    g.add_node("record_turn", record_turn)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")

    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )

    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")

    g.add_edge("refine", "generate")
    g.add_edge("generate", "record_turn")
    g.add_edge("record_turn", END)

    return g


def build_uncompiled_graph() -> StateGraph:
    """Used by scripts/save_graph_image.py - no checkpointer/API keys needed to build this."""
    return _build_uncompiled_graph()


@lru_cache
def build_graph():
    """Singleton compiled graph, checkpointed to SQLite for per-thread short-term memory."""
    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return _build_uncompiled_graph().compile(checkpointer=checkpointer)
