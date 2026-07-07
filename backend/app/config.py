import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(BACKEND_DIR / ".env")

# LangSmith tracing is fully driven by these env vars once set - langchain
# picks them up automatically, no extra client code needed.
os.environ.setdefault("LANGCHAIN_PROJECT", "Clarity CRAG")


class Settings:
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")

    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.environ.get("PINECONE_INDEX_NAME", "clarity-crag")
    pinecone_cloud: str = os.environ.get("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.environ.get("PINECONE_REGION", "us-east-1")

    # Chat model chain: a loop of free, high-volume HuggingFace-hosted open models
    # (routed through HF's Inference Providers) is primary, since Gemini's free tier
    # turned out to be very restrictive (~20 requests/day) for how many LLM calls
    # a single CRAG question makes (grade every doc + filter every strip). Gemini and
    # Groq's Qwen3 stay wired in as the last two fallback tiers after the HF loop.
    hf_chat_models: list[str] = os.environ.get(
        "HF_CHAT_MODELS",
        "meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.1-70B-Instruct",
    ).split(",")
    chat_model: str = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")
    fallback_chat_model: str = os.environ.get("FALLBACK_CHAT_MODEL", "qwen/qwen3-32b")

    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
    fallback_embedding_model: str = os.environ.get(
        "FALLBACK_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-small-en"
    )
    # Both the Gemini (truncated via Matryoshka Representation Learning) and FastEmbed
    # fallback embeddings output 512 dims so they can share one Pinecone index
    # regardless of which one served a given request.
    embedding_dimension: int = int(os.environ.get("EMBEDDING_DIMENSION", 512))

    upper_threshold: float = float(os.environ.get("UPPER_TH", 0.7))
    lower_threshold: float = float(os.environ.get("LOWER_TH", 0.3))
    retriever_k: int = int(os.environ.get("RETRIEVER_K", 4))

    checkpoint_db_path: str = str(DATA_DIR / "checkpoints.db")
    threads_db_path: str = str(DATA_DIR / "threads.db")

    documents_dir: Path = BACKEND_DIR.parent / "documents"
    assets_dir: Path = BACKEND_DIR.parent / "assets"

    cors_origins: list[str] = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()
