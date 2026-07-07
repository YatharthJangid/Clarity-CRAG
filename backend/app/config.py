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
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")

    pinecone_api_key: str = os.environ.get("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.environ.get("PINECONE_INDEX_NAME", "clarity-crag")
    pinecone_cloud: str = os.environ.get("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.environ.get("PINECONE_REGION", "us-east-1")

    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dimension: int = 3072
    chat_model: str = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

    upper_threshold: float = float(os.environ.get("UPPER_TH", 0.7))
    lower_threshold: float = float(os.environ.get("LOWER_TH", 0.3))
    retriever_k: int = int(os.environ.get("RETRIEVER_K", 4))

    checkpoint_db_path: str = str(DATA_DIR / "checkpoints.db")
    threads_db_path: str = str(DATA_DIR / "threads.db")

    documents_dir: Path = BACKEND_DIR.parent / "documents"
    assets_dir: Path = BACKEND_DIR.parent / "assets"

    cors_origins: list[str] = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()
