# ?? RAG Document QA

> Project 5 of the AI Automation Testing Roadmap

A complete Retrieval-Augmented Generation (RAG) pipeline with document ingestion, chunking, embeddings, vector search, retrieval, and LLM-powered answer generation. The project also includes a pytest + DeepEval evaluation suite using Claude as the LLM judge.

**23 tests · 3 DeepEval metrics · Local embeddings · ChromaDB · Claude judge**

---

## ?? Table of Contents

- [Why This Project Exists](#-why-this-project-exists)
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Test Stats](#-test-stats)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Running the Pipeline](#-running-the-pipeline)
- [Running Tests](#-running-tests)
- [Metrics Reference](#-metrics-reference)
- [Configuration](#-configuration)
- [Failure Analysis](#-failure-analysis)
- [Key Concepts](#-key-concepts)
- [How to Extend](#-how-to-extend)
- [Roadmap Progress](#-roadmap-progress)
- [Expected Test Results](#-expected-test-results)
- [Important Notes](#-important-notes)
- [Author](#-author)
- [License](#-license)

---

## ?? Why This Project Exists

In Projects 1–4, we tested LLM and chatbot applications where the context came from the prompt or the model's own knowledge.

Modern AI applications often use Retrieval-Augmented Generation (RAG): they fetch relevant context from a knowledge base before generating an answer. This changes how we test software:

- Retrieval can fail when the wrong document is selected.
- Context can be noisy when relevant and irrelevant details are mixed together.
- Generation can hallucinate information not grounded in the retrieved context.
- Faithfulness matters because every answer should be supported by the source material.

> ### ?? Core Principle
>
> RAG quality = retrieval quality × generation quality.

A perfect model is not enough if the retrieval layer returns the wrong context. This project builds both the pipeline and the tests needed to detect those failures.

---

## ?? Overview

This project builds a RAG system for an e-commerce customer support knowledge base with:

- ? Document ingestion for `.txt` policy files
- ? Chunking with overlapping windows
- ? Local embeddings using `all-MiniLM-L6-v2`
- ? Persistent vector search with ChromaDB
- ? Semantic retrieval with top-K search
- ? Claude-powered grounded answer generation
- ? Refusal behavior for out-of-domain questions
- ? 23 automated tests for retrieval, keyword checks, refusal logic, and DeepEval metrics
- ? Claude used as the LLM judge without requiring an OpenAI key

---

## ??? Architecture

```text
Documents --> Chunking --> Embeddings --> Vector DB --> Retriever --> Context --> LLM --> Answer
     (data/documents/*.txt)     (rag/ingest.py)   (SentenceTransformer)   (ChromaDB)     (rag/retriever.py)
```

---

## ?? Test Stats

| Test Suite | Tests | Purpose |
| --- | ---: | --- |
| `test_rag_pipeline.py` | 11 | Retrieval, keyword checks, and refusal behavior |
| `test_rag_deepeval.py` | 12 | Answer relevancy, faithfulness, and contextual relevancy |
| **Total** | **23** | **End-to-end RAG validation** |

### Test breakdown

| Test | Count | What it checks |
| --- | ---: | --- |
| `test_retrieval_finds_expected_source` | 4 | Whether the retriever finds the right document |
| `test_answer_contains_expected_keywords` | 4 | Whether the answer includes key facts |
| `test_out_of_domain_refusal` | 3 | Whether the system refuses unknown questions |
| `test_rag_answer_relevancy` | 4 | Whether the answer is relevant |
| `test_rag_faithfulness` | 4 | Whether the answer is grounded in context |
| `test_rag_contextual_relevancy` | 4 | Whether the retrieved context is useful |

---

## ??? Project Structure

```text
rag-document-qa/
+-- data/
¦   +-- documents/
¦   ¦   +-- return-policy.txt
¦   ¦   +-- shipping-policy.txt
¦   ¦   +-- payment-policy.txt
¦   ¦   +-- warranty-policy.txt
¦   +-- chroma_db/                 # Persistent vector store
+-- rag/
¦   +-- __init__.py
¦   +-- ingest.py                  # Load and chunk documents
¦   +-- vectorstore.py             # ChromaDB wrapper
¦   +-- retriever.py               # Semantic search
¦   +-- generator.py               # Claude answer generation
¦   +-- pipeline.py                # End-to-end RAG flow
+-- config/
¦   +-- __init__.py
¦   +-- settings.py                # Chunk size, TOP_K, model names
+-- tests/
¦   +-- __init__.py
¦   +-- conftest.py                # Fixtures
¦   +-- deepeval_judge.py          # Claude judge
¦   +-- test_rag_pipeline.py       # 11 pipeline tests
¦   +-- test_rag_deepeval.py       # 12 DeepEval tests
+-- reports/
¦   +-- rag_report.html            # Generated HTML report
+-- .env
+-- .env.example
+-- .gitignore
+-- pyproject.toml
+-- requirements.txt
+-- README.md
+-- LICENSE
+-- test_data/
    +-- failures.md               # Failure-analysis notes
```

---

## ??? Tech Stack

| Technology | Purpose |
| --- | --- |
| **Python 3.11+** | Application and test code |
| **ChromaDB** | Persistent vector database |
| **SentenceTransformers** | Local embeddings using `all-MiniLM-L6-v2` |
| **Anthropic SDK** | Claude API for generation and judging |
| **DeepEval 4.2+** | RAG evaluation metrics |
| **pytest** | Test automation |
| **python-dotenv** | Environment configuration |
| **pytest-timeout** | Per-test timeouts |
| **pytest-rerunfailures** | Retry support for flaky tests |

**No OpenAI key required.** Embeddings run locally; the judge runs on Claude.

---

## ?? Prerequisites

- Python 3.11+
- pip
- Git
- An **Anthropic API key** with available credits
- ~500 MB of free disk space for the embedding model and ChromaDB
- Terminal or PowerShell access

---

## ?? Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
notepad .env
```

Add your Anthropic key:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-real-key-here
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

> Never commit `.env` to Git.

### 5. Build the RAG index

```bash
python -m rag.ingest
python -m rag.vectorstore
```

The first run downloads the embedding model (about 80 MB), which is then cached locally.

### 6. Test the pipeline

```bash
python -m rag.pipeline
```

Expected output: 5 Q&A pairs with source references.

### 7. Run the tests

```bash
pytest tests/ -v --timeout=240
```

---

## ?? Running the Pipeline

All pipeline commands use `python -m`:

```bash
# Load and chunk documents
python -m rag.ingest

# Build the vector store
python -m rag.vectorstore

# Test retrieval only
python -m rag.retriever

# Test generation only
python -m rag.generator

# Full end-to-end pipeline
python -m rag.pipeline
```

### Example output

```text
Q: What is your return policy?
A: We accept returns within 30 days of purchase for a full refund...
Sources: ['return-policy.txt', 'return-policy.txt', 'return-policy.txt']

Q: Do you ship to Mars?
A: I don't have enough information to answer that.
Sources: ['shipping-policy.txt', ...]
```

---

## ?? Running Tests

### Run all tests

```bash
pytest tests/ -v --timeout=240
```

### Pipeline tests only

```bash
pytest tests/test_rag_pipeline.py -v --timeout=240
```

### DeepEval tests only

```bash
pytest tests/test_rag_deepeval.py -v --timeout=240
```

### One specific test

```bash
pytest "tests/test_rag_deepeval.py::test_rag_answer_relevancy[What is your return policy?]" -v --timeout=120
```

### HTML report

```bash
pytest tests/ -v --html=reports/rag_report.html --self-contained-html --timeout=240
start reports/rag_report.html
```

---

## ?? Metrics Reference

### Retrieval metrics

| Metric | Direction | Threshold | Purpose |
| --- | --- | --- | --- |
| `test_retrieval_finds_expected_source` | Boolean | Must match | Did the retriever find the right document? |
| `test_answer_contains_expected_keywords` | Boolean | Must contain | Are the key facts present in the answer? |

### DeepEval RAG metrics

| Metric | Direction | Threshold | Purpose |
| --- | --- | --- | --- |
| `AnswerRelevancyMetric` | Higher = better | = 0.70 | Did the answer address the question? |
| `FaithfulnessMetric` | Higher = better | = 0.70 | Is the answer grounded in context? |
| `ContextualRelevancyMetric` | Higher = better | = 0.70 | Is the retrieved context relevant? |

> DeepEval 4.x behavior: a score passes when `score >= threshold`.

---

## ?? Configuration

All settings live in `config/settings.py`:

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
```

### Chunk size tuning

| Size | Use case | Trade-off |
| --- | --- | --- |
| 150–300 | Small policy snippets | Focused, but answers may split |
| 300–500 | Medium documents | Balance of recall and precision |
| 500–1000 | Long articles | Better recall, more noise |
| 1000+ | Books/manuals | High recall, lower precision |

For policy documents, a range of 200–300 characters often works best.

---

## ?? Failure Analysis

Failures are documented in `test_data/failures.md`.

### Known issue: contextual relevancy

**Symptoms:**

- “How long does shipping take?”
- “What payment methods do you accept?”
- “What is your warranty policy?”

**Judge reason:**

> The retrieved context contains the answer but is diluted by irrelevant details about cost, coverage, tracking, and other policies.

**Root cause:**

The original chunk size was 500 characters. Because each policy document is around 500–615 characters, most files produced only one chunk. The retriever then returned the entire file, including unrelated content.

**Fix:**

- Reduce `CHUNK_SIZE` to 200
- Reduce `CHUNK_OVERLAP` to 50
- Delete `data/chroma_db/`
- Rebuild the index

**Status:** ? Documented and fixed.

This is a real RAG quality issue, not a test bug: smaller chunks improve precision but can reduce recall if answers are split across chunks.

---

## ?? Key Concepts

### 1. Retrieval-Augmented Generation

RAG = retrieve relevant documents ? augment the prompt ? generate the answer.

Benefits:

- Grounded answers with less hallucination
- Updatable knowledge without retraining
- Citable sources for auditability

### 2. Chunking Strategy

Chunking is one of the most important decisions in a RAG pipeline:

- Too small: answers can split across chunks and recall drops.
- Too large: answers get buried in noise and precision drops.
- Overlap helps prevent useful context from being lost at chunk boundaries.

### 3. Local Embeddings

`all-MiniLM-L6-v2` is:

- Free and local
- Fast on CPU
- Small (~80 MB)
- Good enough for many semantic search tasks

For production systems, larger models such as `bge-large` or `e5-large` may be preferred.

### 4. ChromaDB

An embedded vector database:

- Persistent across restarts
- No separate server required
- Uses cosine similarity for text retrieval
- Supports metadata filtering by source or tag

### 5. RAG metrics

DeepEval provides 3 key metrics:

- Answer Relevancy — did the answer address the question?
- Faithfulness — are all claims grounded in the retrieved context?
- Contextual Relevancy — was the retrieved context actually useful?

All 3 should pass for a healthy RAG system.

### 6. Claude as judge

No OpenAI key is required. The custom `ClaudeJudge` extends `DeepEvalBaseLLM` and passes `model=judge` to every metric.

---

## ?? How to Extend

### Exercise 1 — Add more documents

Drop more `.txt` files into `data/documents/` and rebuild the index:

```bash
python -m rag.ingest
python -m rag.vectorstore
```

### Exercise 2 — Try different chunk sizes

Edit `config/settings.py`, then:

```powershell
Remove-Item -Recurse -Force data/chroma_db
python -m rag.ingest
python -m rag.vectorstore
python -m rag.pipeline
```

Compare contextual relevancy across chunk sizes.

### Exercise 3 — Add PDF support

Install `pypdf` and extend `rag/ingest.py` to load `.pdf` files.

### Exercise 4 — Add metadata filtering

```python
collection.query(
    query_texts=[query],
    where={"source": "return-policy.txt"},
    n_results=5,
)
```

### Exercise 5 — Add reranking

Retrieve top-20 chunks and rerank them to the top-3 using a cross-encoder.

### Exercise 6 — Add hybrid search

Combine BM25 keyword search with embedding-based semantic retrieval.

### Exercise 7 — Multi-query RAG

Generate multiple query variants, retrieve for each one, and merge the results.

---

## ??? Roadmap Progress

| Project | Status |
| --- | --- |
| Project 1 — LLM Testing Fundamentals | ? |
| Project 2 — DeepEval LLM Evaluation | ? |
| Project 3 — Golden Dataset | ? |
| Project 4 — AI Chatbot Testing Framework | ? |
| Project 5 — RAG Document QA | ? You are here |
| Project 6 — RAG Testing with DeepEval | ? |
| Project 7 — AI Agent Testing with DeepEval | ? |
| Project 8 — AI Evaluation Automation Framework | ? |
| Project 9 — AI Regression Testing | ? |
| Project 10 — AI Cost & Performance Testing | ? |
| Project 11 — AI Adversarial Testing | ? |
| Project 12 — Production AI Quality Monitoring | ? |
| Project 13 — AI Testing CI/CD | ? |
| Project 14 — Production AI QA Platform | ? |

---

## ?? Expected Test Results

After tuning chunk size to 200, the expected result is:

```text
collected 23 items

tests/test_rag_deepeval.py::test_rag_answer_relevancy[...] PASSED
tests/test_rag_deepeval.py::test_rag_faithfulness[...] PASSED
tests/test_rag_deepeval.py::test_rag_contextual_relevancy[...] PASSED
tests/test_rag_pipeline.py::test_retrieval_finds_expected_source[...] PASSED
tests/test_rag_pipeline.py::test_answer_contains_expected_keywords[...] PASSED
tests/test_rag_pipeline.py::test_out_of_domain_refusal[...] PASSED

============================= 23 passed in ~3 min =============================
```

> Runtime and scores can vary because both generation and the judge make real LLM calls.

---

## ?? Important Notes

### ?? API security

Never commit `.env` files. Rotate your API keys immediately if they are exposed.

### ?? Non-determinism

LLM judges can vary. The `@pytest.mark.flaky(reruns=2)` marker retries flaky failures automatically.

### ?? Threshold calibration

Do not lower thresholds just to pass tests. Diagnose the cause:

- Retrieval issue ? tune chunk size or `TOP_K`
- Generation issue ? refine the system prompt
- Test issue ? fix the test data or metric logic

### ????? Judge quality

Validate LLM judges against human labels for high-risk systems.

---

## ?? Author

Your Name

- GitHub: `@your-username`
- LinkedIn: Your Name

Project 5 of 14 — AI Automation Testing Roadmap

---

## ?? License

MIT License — see `LICENSE` for details.

---

## ? If This Project Helped You

Give it a star on GitHub and follow the roadmap as it progresses from basic LLM testing to a production-grade AI QA platform.

---

## ?? Supporting Files to Create

### `requirements.txt`

```txt
chromadb>=0.5.0
sentence-transformers>=3.0.0
anthropic>=0.40.0
deepeval>=4.2.0
pytest>=8.0.0
pytest-timeout>=2.0.0
pytest-rerunfailures>=16.0.0
python-dotenv>=1.0.0
pypdf>=4.0.0
```

### `.env.example`

```env
ANTHROPIC_API_KEY=your_api_key_here
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

### `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc
.env
data/chroma_db/
reports/
.pytest_cache/
.cache/
```

### `pyproject.toml`

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning:deepeval.*",
    "ignore::UserWarning:huggingface_hub.*",
    "ignore::UserWarning:sentence_transformers.*",
]
```

### `LICENSE`

Use a standard MIT license with your name and year.
