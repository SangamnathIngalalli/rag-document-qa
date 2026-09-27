"""RAG pipeline configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Chunking settings
CHUNK_SIZE = 200          # characters per chunk
CHUNK_OVERLAP = 50       # overlap between chunks

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, local, 384-dim

# Retrieval settings
TOP_K = 3                 # number of chunks to retrieve

# LLM settings
LLM_MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024