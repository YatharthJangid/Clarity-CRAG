from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import settings
from app.crag.graph import build_graph
from app.schemas import (
    MessageOut,
    SendMessageIn,
    SendMessageOut,
    ThreadDetailOut,
    ThreadOut,
)

app = FastAPI(title="Clarity CRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _blank_turn_input(question: str) -> dict:
    return {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/threads", response_model=ThreadOut)
def create_thread():
    return db.create_thread()


@app.get("/api/threads", response_model=list[ThreadOut])
def list_threads():
    return db.list_threads()


@app.get("/api/threads/{thread_id}", response_model=ThreadDetailOut)
def get_thread(thread_id: str):
    thread = db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = build_graph().get_state(config)
    history = snapshot.values.get("history", []) if snapshot and snapshot.values else []

    return {"thread_id": thread_id, "title": thread["title"], "messages": history}


@app.post("/api/threads/{thread_id}/messages", response_model=SendMessageOut)
def send_message(thread_id: str, body: SendMessageIn):
    thread = db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    config = {"configurable": {"thread_id": thread_id}}
    result = build_graph().invoke(_blank_turn_input(body.content), config=config)
    db.touch_thread(thread_id, first_message=body.content)

    return {
        "answer": result["answer"],
        "verdict": result["verdict"],
        "reason": result["reason"],
        "web_query": result.get("web_query", ""),
        "used_web_search": result["verdict"] != "CORRECT",
    }
