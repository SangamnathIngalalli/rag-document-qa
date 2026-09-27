"""Retrieval: find relevant chunks for a query."""

from typing import List, Dict
from rag.vectorstore import get_client, get_or_create_collection
from config.settings import TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Returns a list of dicts with: text, source, distance, id
    """
    client = get_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })

    return chunks


if __name__ == "__main__":
    query = "What is your return policy?"
    results = retrieve(query)

    print(f"Query: {query}\n")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['source']}] (distance: {r['distance']:.4f})")
        print(f"   {r['text'][:200]}...\n")