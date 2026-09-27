"""LLM answer generation from retrieved context."""

import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from config.settings import LLM_MODEL, MAX_TOKENS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful customer support assistant for an e-commerce store.

Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say: "I don't have enough information to answer that."

Do NOT use any knowledge outside the provided context.
Be concise and direct."""


def generate_answer(query: str, context_chunks: list) -> str:
    """
    Generate an answer from the query and retrieved context.
    """
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )


if __name__ == "__main__":
    from rag.retriever import retrieve

    query = "What is your return policy?"
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)

    print(f"Query: {query}\n")
    print(f"Answer: {answer}\n")
    print("Sources used:")
    for c in chunks:
        print(f"  - {c['source']}")