"""End-to-end RAG pipeline."""

from typing import Dict
from rag.retriever import retrieve
from rag.generator import generate_answer


def ask(query: str, top_k: int = 3) -> Dict:
    """
    Run the full RAG pipeline: retrieve → generate.

    Returns a dict with: query, answer, retrieved_chunks, sources
    """
    chunks = retrieve(query, top_k=top_k)
    answer = generate_answer(query, chunks)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": [c["text"] for c in chunks],
        "sources": [c["source"] for c in chunks],
        "distances": [c["distance"] for c in chunks],
    }


if __name__ == "__main__":
    import json

    test_questions = [
        "What is your return policy?",
        "How long does shipping take?",
        "What payment methods do you accept?",
        "What is your warranty policy?",
        "Do you ship to Mars?",
    ]

    for q in test_questions:
        result = ask(q)
        print("=" * 70)
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print()