"""
RAG pipeline tests.

Covers:
- Retrieval quality (does the retriever find relevant docs?)
- Answer relevance (does the answer address the question?)
- Groundedness (is the answer faithful to context?)
- Refusal behavior (does it say "I don't know" for out-of-domain?)
"""

import pytest


TEST_QUESTIONS = [
    {
        "query": "What is your return policy?",
        "expected_source": "return-policy.txt",
        "expected_keywords": ["30 days", "full refund"],
    },
    {
        "query": "How long does shipping take?",
        "expected_source": "shipping-policy.txt",
        "expected_keywords": ["5-7", "business days"],
    },
    {
        "query": "What payment methods do you accept?",
        "expected_source": "payment-policy.txt",
        "expected_keywords": ["Visa", "PayPal"],
    },
    {
        "query": "What is your warranty policy?",
        "expected_source": "warranty-policy.txt",
        "expected_keywords": ["1-year", "manufacturer"],
    },
]

OUT_OF_DOMAIN_QUESTIONS = [
    "What is the capital of France?",
    "How do I make a cake?",
    "Who won the World Cup?",
]


@pytest.mark.parametrize("case", TEST_QUESTIONS, ids=[c["query"] for c in TEST_QUESTIONS])
def test_retrieval_finds_expected_source(rag_retrieve, case):
    """The retriever should find the document containing the answer."""
    results = rag_retrieve(case["query"], top_k=3)
    sources = [r["source"] for r in results]
    assert case["expected_source"] in sources, (
        f"Expected {case['expected_source']} in retrieved sources, got {sources}"
    )


@pytest.mark.parametrize("case", TEST_QUESTIONS, ids=[c["query"] for c in TEST_QUESTIONS])
def test_answer_contains_expected_keywords(rag_ask, case):
    """The answer should contain expected keywords from the source."""
    result = rag_ask(case["query"])
    answer_lower = result["answer"].lower()
    for keyword in case["expected_keywords"]:
        assert keyword.lower() in answer_lower, (
            f"Expected '{keyword}' in answer: {result['answer']}"
        )


@pytest.mark.parametrize("query", OUT_OF_DOMAIN_QUESTIONS)
def test_out_of_domain_refusal(rag_ask, query):
    """The RAG system should refuse out-of-domain questions."""
    result = rag_ask(query)
    answer_lower = result["answer"].lower()
    refusal_phrases = ["don't have", "do not have", "not enough information", "cannot answer"]
    assert any(phrase in answer_lower for phrase in refusal_phrases), (
        f"Expected refusal for '{query}', got: {result['answer']}"
    )