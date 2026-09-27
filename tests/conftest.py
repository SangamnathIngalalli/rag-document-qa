import pytest
from rag.pipeline import ask
from rag.retriever import retrieve


@pytest.fixture(scope="session")
def rag_ask():
    """Provide the RAG ask function."""
    return ask


@pytest.fixture(scope="session")
def rag_retrieve():
    """Provide the retriever."""
    return retrieve