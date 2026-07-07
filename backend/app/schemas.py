from typing import Optional

from pydantic import BaseModel


class ThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    verdict: Optional[str] = None
    used_web_search: Optional[bool] = None


class ThreadDetailOut(BaseModel):
    thread_id: str
    title: str
    messages: list[MessageOut]


class SendMessageIn(BaseModel):
    content: str


class SendMessageOut(BaseModel):
    answer: str
    verdict: str
    reason: str
    web_query: str
    used_web_search: bool
