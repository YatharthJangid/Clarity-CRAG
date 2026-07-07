from functools import lru_cache

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


class FallbackEmbeddings(Embeddings):
    """Tries Gemini embeddings first, drops to a local FastEmbed model on error.

    Both are configured for `settings.embedding_dimension`-dim vectors so either one
    can serve a request against the same Pinecone index. Gemini's `output_dimensionality`
    only takes effect when passed to the embed calls themselves - passing it to the
    constructor is silently ignored - so it's threaded through here on every call.
    """

    def __init__(self, primary: GoogleGenerativeAIEmbeddings, fallback: Embeddings):
        self.primary = primary
        self.fallback = fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts, output_dimensionality=settings.embedding_dimension)
        except Exception:
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.primary.embed_query(text, output_dimensionality=settings.embedding_dimension)
        except Exception:
            return self.fallback.embed_query(text)


_primary_embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.gemini_api_key)
_fallback_embeddings = FastEmbedEmbeddings(model_name=settings.fallback_embedding_model)
embeddings = FallbackEmbeddings(_primary_embeddings, _fallback_embeddings)

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
    return PineconeVectorStore(index_name=settings.pinecone_index_name, embedding=embeddings)


@lru_cache
def get_retriever():
    """Lazily built + cached: touches the network (Pinecone) on first real use only,
    so importing this module never requires live credentials."""
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": settings.retriever_k})
