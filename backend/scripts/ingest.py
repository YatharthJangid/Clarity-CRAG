"""Loads the PDFs in documents/, splits them, and upserts them into Pinecone.

Replaces the notebook's `FAISS.from_documents(chunks, embeddings)` with a
Pinecone-backed index so the vector store is durable and shared by the API.

Usage: python backend/scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.crag.vectorstore import ensure_index, get_vector_store


def load_and_split() -> list:
    pdf_paths = sorted(settings.documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {settings.documents_dir}")

    docs = []
    for path in pdf_paths:
        docs.extend(PyPDFLoader(str(path)).load())

    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return chunks


def main():
    print(f"Loading PDFs from {settings.documents_dir} ...")
    chunks = load_and_split()
    print(f"Split into {len(chunks)} chunks.")

    print(f"Ensuring Pinecone index '{settings.pinecone_index_name}' exists ...")
    ensure_index()

    print("Upserting chunks into Pinecone (this calls Gemini embeddings for each chunk) ...")
    vector_store = get_vector_store()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"  upserted {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Done. Index is ready for retrieval.")


if __name__ == "__main__":
    main()
