"""
config.py — Configurazione centralizzata del progetto RAG
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# WebBaseLoader requires a user agent
os.environ["USER_AGENT"] = "rag-project/1.0"

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

# Crea le directory se non esistono
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys e Modelli ────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 1000       # Caratteri per chunk
CHUNK_OVERLAP: int = 150     # Overlap tra chunk contigui

# ── Retrieval ───────────────────────────────────────────────────────────────
TOP_K: int = 5               # Numero di chunk da recuperare per query

# ── Chroma ──────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "rag_docs"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: Path = BASE_DIR / "logs"
