import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str

    # Accumulates across turns for the same thread_id via the SqliteSaver
    # checkpointer - this is what gives each thread its short-term memory.
    history: Annotated[List[dict], operator.add]
