"""RAG evaluation with DeepEval metrics using Claude as judge."""

import os
from pathlib import Path
import pytest
from dotenv import load_dotenv
from anthropic import Anthropic
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import DeepEvalBaseLLM

from rag.pipeline import ask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="claude-sonnet-5"):
        self.model_name = model_name
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


judge = ClaudeJudge()


EVAL_CASES = [
    {
        "query": "What is your return policy?",
        "expected_output": "Returns are accepted within 30 days of purchase for a full refund.",
    },
    {
        "query": "How long does shipping take?",
        "expected_output": "Standard shipping takes 5-7 business days.",
    },
    {
        "query": "What payment methods do you accept?",
        "expected_output": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
    },
    {
        "query": "What is your warranty policy?",
        "expected_output": "Electronics come with a 1-year manufacturer warranty.",
    },
]


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["query"] for c in EVAL_CASES])
def test_rag_answer_relevancy(case):
    result = ask(case["query"])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=result["answer"],
        expected_output=case["expected_output"],
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["query"] for c in EVAL_CASES])
def test_rag_faithfulness(case):
    result = ask(case["query"])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=result["answer"],
        retrieval_context=result["retrieved_chunks"],
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["query"] for c in EVAL_CASES])
def test_rag_contextual_relevancy(case):
    result = ask(case["query"])

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=result["answer"],
        retrieval_context=result["retrieved_chunks"],
    )

    metric = ContextualRelevancyMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])