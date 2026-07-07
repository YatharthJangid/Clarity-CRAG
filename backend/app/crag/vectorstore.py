from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

_embeddings = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
_pc = Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    """Create the Pinecone index if it doesn't exist yet. Called by the ingest script."""
    if settings.pinecone_index_name not in [i["name"] for i in _pc.list_indexes()]:
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )


def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(index_name=settings.pinecone_index_name, embedding=_embeddings)


@lru_cache
def get_retriever():
    """Lazily built + cached: touches the network (Pinecone) on first real use only,
    so importing this module never requires live credentials."""
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": settings.retriever_k})
