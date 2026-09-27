"""ChromaDB vector store management."""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from config.settings import CHROMA_DIR, EMBEDDING_MODEL


def get_client():
    """Get or create a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_embedding_function():
    """Use a local SentenceTransformer model for embeddings (free, no API)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_or_create_collection(client=None, name: str = "rag_documents"):
    """Get or create the collection."""
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: List[Dict]):
    """Upsert chunks into the collection (idempotent)."""
    if not chunks:
        return

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "source": c["source"],
            "chunk_index": c["chunk_index"],
        } for c in chunks],
    )


if __name__ == "__main__":
    from rag.ingest import load_documents, prepare_chunks

    docs = load_documents()
    chunks = prepare_chunks(docs)

    client = get_client()
    collection = get_or_create_collection(client)

    upsert_chunks(collection, chunks)
    print(f"✅ Upserted {len(chunks)} chunks into ChromaDB")
    print(f"   Collection count: {collection.count()}")