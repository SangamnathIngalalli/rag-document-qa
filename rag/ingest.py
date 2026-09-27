"""Document loading and chunking."""

from pathlib import Path
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_documents(data_dir: Path = DATA_DIR) -> List[Dict]:
    """Load all .txt files from the data directory."""
    documents = []
    for path in sorted(data_dir.glob("*.txt")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "id": path.stem,
            "source": path.name,
            "text": text,
        })
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.

    Overlap helps maintain context across chunk boundaries.
    """
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if (end - overlap) > start else end

    return chunks


def prepare_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Convert documents into chunked records ready for embedding.

    Each record has: id, source, text, chunk_index
    """
    chunked = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked.append({
                "id": f"{doc['id']}-chunk-{i:03d}",
                "source": doc["source"],
                "text": chunk,
                "chunk_index": i,
            })
    return chunked


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents:")
    for d in docs:
        print(f"  - {d['source']} ({len(d['text'])} chars)")

    chunks = prepare_chunks(docs)
    print(f"\nCreated {len(chunks)} chunks:")
    for c in chunks[:5]:
        print(f"  - {c['id']} ({len(c['text'])} chars)")