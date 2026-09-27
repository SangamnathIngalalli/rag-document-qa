# 🤖 Project 05 — RAG Document Question Answering

> **AI Automation Testing Portfolio — Project 05**

A hands-on **Retrieval-Augmented Generation (RAG)** application that answers questions from an e-commerce knowledge base using document chunking, local embeddings, ChromaDB retrieval, and Claude-based answer generation.

The project also includes an automated **Pytest + DeepEval** evaluation suite for validating retrieval, answer content, out-of-domain refusal behavior, answer relevancy, faithfulness, and contextual relevancy.

---

## 🎯 Project Objective

This project demonstrates how to build and test a RAG pipeline where the LLM is grounded in retrieved business documents rather than relying only on its general knowledge.

The goal is to validate two important layers of a RAG application:

- **Retrieval quality** — does the system retrieve the right source information?
- **Generation quality** — does the LLM produce a relevant and grounded answer from that context?

---

## 🔄 RAG Pipeline

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Local Embeddings
    ↓
ChromaDB Vector Store
    ↓
Semantic Retriever
    ↓
Retrieved Context
    ↓
Claude LLM
    ↓
Grounded Answer
```

The application uses a local SentenceTransformers embedding model and ChromaDB for vector search, while Claude is used to generate the final answer.

---

## 🧪 Testing & Evaluation

The project contains **23 parametrized automated test cases** across two test modules.

| Test Module | Cases | Purpose |
|---|---:|---|
| `test_rag_pipeline.py` | 11 | Retrieval, expected answer content, and out-of-domain refusal |
| `test_rag_deepeval.py` | 12 | DeepEval evaluation of answer relevancy, faithfulness, and contextual relevancy |
| **Total** | **23** | **RAG application validation** |

### Functional / Pipeline Validation

The pipeline tests verify:

- Expected source documents are retrieved
- Answers contain important expected facts
- Out-of-domain questions receive a refusal response

The current test data covers:

- Return policy
- Shipping policy
- Payment methods
- Warranty policy
- Out-of-domain questions

### DeepEval Evaluation

The DeepEval suite evaluates four knowledge-base questions using:

- **Answer Relevancy** — whether the generated answer addresses the question
- **Faithfulness** — whether the answer is supported by retrieved context
- **Contextual Relevancy** — whether the retrieved context is relevant to the question

The evaluation threshold configured in the tests is **0.70**.

---

## 📚 Knowledge Base

The current document set contains four e-commerce policy documents:

```text
data/documents/
├── return-policy.txt
├── shipping-policy.txt
├── payment-policy.txt
└── warranty-policy.txt
```

Examples of information covered include:

- Return windows and refunds
- Standard and express shipping
- Accepted payment methods
- Warranty coverage

The generator is explicitly instructed to answer **only from the retrieved context** and to respond with:

> “I don't have enough information to answer that.”

when the supplied context does not contain the answer.

---

## 🏗️ Project Structure

```text
rag-document-qa/
│
├── config/
│   └── settings.py              # Pipeline configuration
│
├── data/
│   └── documents/               # Knowledge-base documents
│       ├── return-policy.txt
│       ├── shipping-policy.txt
│       ├── payment-policy.txt
│       └── warranty-policy.txt
│
├── rag/
│   ├── ingest.py                # Document loading & chunking
│   ├── vectorstore.py            # ChromaDB & embeddings
│   ├── retriever.py              # Semantic retrieval
│   ├── generator.py              # Claude answer generation
│   └── pipeline.py               # End-to-end RAG flow
│
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── deepeval_judge.py         # Claude DeepEval judge
│   ├── test_rag_pipeline.py      # Pipeline validation
│   └── test_rag_deepeval.py      # RAG quality evaluation
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🧩 Core Components

### 1. Document Ingestion

`rag/ingest.py`:

- Loads `.txt` documents from the knowledge base
- Creates document records with source metadata
- Splits documents into overlapping chunks
- Assigns an ID and chunk index to each chunk

Current configuration:

```text
Chunk size:     200 characters
Chunk overlap:   50 characters
```

### 2. Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

through SentenceTransformers.

The embedding model runs locally, so an embedding API key is not required.

### 3. Vector Database

**ChromaDB** is used as a persistent vector store.

The implementation:

- Creates a persistent ChromaDB client
- Uses cosine similarity
- Stores document chunks and source metadata
- Uses idempotent upserts

### 4. Retrieval

The retriever performs semantic search against the ChromaDB collection.

Current configuration:

```text
TOP_K = 3
```

The retrieved records include:

- Chunk ID
- Chunk text
- Source document
- Similarity distance

### 5. Answer Generation

Claude receives:

- The user question
- The retrieved context
- Source information for the retrieved chunks

The system prompt instructs Claude to:

- Use only the supplied context
- Avoid outside knowledge
- Keep answers concise
- Refuse questions when the context does not contain the answer

### 6. LLM-as-a-Judge

DeepEval is configured with a custom `ClaudeJudge` implementation based on `DeepEvalBaseLLM`.

This allows the project to use Claude for evaluating RAG responses without requiring an OpenAI API key.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Application and test automation |
| **Pytest** | Automated testing |
| **DeepEval** | LLM/RAG evaluation |
| **Anthropic Claude** | Answer generation and LLM-as-a-Judge |
| **SentenceTransformers** | Local text embeddings |
| **all-MiniLM-L6-v2** | Embedding model |
| **ChromaDB** | Persistent vector database |
| **python-dotenv** | Environment configuration |

### Key Testing Concepts

- RAG pipeline testing
- Retrieval validation
- Semantic evaluation
- LLM-as-a-Judge
- Faithfulness testing
- Contextual relevancy
- Out-of-domain testing
- Grounded answer validation

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- pip
- Git
- Anthropic API key with available credits

### 1. Clone the repository

```bash
git clone https://github.com/SangamnathIngalalli/rag-document-qa.git
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

### 4. Configure the API key

Create a local `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Do not commit `.env` to Git.

---

## ▶️ Build the RAG Index

Load and chunk the documents:

```bash
python -m rag.ingest
```

Build/update the ChromaDB vector store:

```bash
python -m rag.vectorstore
```

The embedding model is downloaded locally on first use and then cached by the underlying library.

---

## 💬 Run the RAG Pipeline

Run the end-to-end example:

```bash
python -m rag.pipeline
```

The example asks questions about:

- Return policy
- Shipping
- Payment methods
- Warranty
- An out-of-domain question

The pipeline prints the generated answer and the source documents returned by the retriever.

---

## 🔎 Test Individual Components

### Test retrieval

```bash
python -m rag.retriever
```

### Test answer generation

```bash
python -m rag.generator
```

### Run all automated tests

```bash
pytest tests/ -v
```

### Run pipeline tests only

```bash
pytest tests/test_rag_pipeline.py -v
```

### Run DeepEval tests only

```bash
pytest tests/test_rag_deepeval.py -v
```

> DeepEval tests make real LLM judge calls, so execution time and evaluation behavior can vary.

---

## 📊 Evaluation Model

The project separates **application behavior checks** from **LLM-based semantic evaluation**.

```text
                    RAG Application
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
       Pipeline Tests            DeepEval Tests
              │                       │
              ↓                       ↓
      Source validation        Answer Relevancy
      Keyword validation       Faithfulness
      Refusal validation       Contextual Relevancy
```

This gives the project both deterministic checks and semantic quality evaluation.

---

## ⚙️ Configuration

Current settings are defined in `config/settings.py`:

```python
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3

LLM_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
```

These parameters make the RAG pipeline easy to experiment with, particularly around chunking and retrieval behavior.

---

## 🔬 Key Learning Areas

### Retrieval is part of AI quality

A RAG system can produce a poor answer even when the underlying LLM is capable if the retriever supplies the wrong or insufficient context.

### Chunking affects retrieval

Chunk size and overlap influence how information is represented in the vector store and what context reaches the LLM.

### Grounding needs explicit evaluation

Checking whether an answer sounds correct is not enough. The project uses faithfulness evaluation to test whether the generated response is supported by retrieved context.

### Deterministic and semantic tests complement each other

Traditional assertions are useful for checking known facts and system behavior, while LLM-based metrics help evaluate qualities that are difficult to capture with exact string assertions.

---

## 🧪 Project Testing Progression

This project extends the testing concepts developed in the earlier portfolio projects:

```text
Project 01
LLM Functional Testing
        ↓
Project 02
LLM Semantic Evaluation
        ↓
Project 03
Golden Dataset Engineering
        ↓
Project 04
Multi-Turn Chatbot Testing
        ↓
Project 05
RAG Application Testing & Evaluation
```

Project 05 introduces a new testing dimension: **the interaction between retrieval quality, supplied context, and generated answers**.

---

## 🎯 Future Extensions

Potential next steps for the RAG application include:

- PDF document ingestion
- Metadata-based filtering
- Hybrid keyword + semantic search
- Retrieval reranking
- Multi-query retrieval
- Larger or alternative embedding models
- More extensive RAG evaluation datasets
- Retrieval-specific metrics
- CI/CD execution for RAG tests

These are future extensions rather than claims about the current implementation.

---

## 👨‍💻 Author

**Sangamnath Ingalalli**

AI / LLM Testing | Test Automation | LLM Evaluation | Quality Engineering | Python Automation | AI QA

### 🔗 Links

- 💼 [LinkedIn](https://www.linkedin.com/in/sangamnath-ingalalli-a4b954115/)
- 🐙 [GitHub](https://github.com/SangamnathIngalalli)
- 📂 [AI Automation Testing Portfolio](https://github.com/SangamnathIngalalli/AI-Automation-Testing-Portfolio)

---

## 📌 Project Summary

**Project 05 — RAG Document Question Answering**

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Context
   ↓
Claude
   ↓
Answer
   ↓
Pytest + DeepEval Evaluation
```

> **Focus:** Building and testing a grounded RAG application using retrieval validation, deterministic assertions, and LLM-based evaluation.
