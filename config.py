"""
config.py — Central configuration for Legal Contract Q&A RAG system
All paths, model names, and hyperparameters live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_RAW_DIR    = BASE_DIR / "data" / "raw"
DATA_PROC_DIR   = BASE_DIR / "data" / "processed"
VECTORDB_DIR    = BASE_DIR / "vectordb"
EVAL_DIR        = BASE_DIR / "evaluation"

for d in [DATA_RAW_DIR, DATA_PROC_DIR, VECTORDB_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~80MB, fast, good for semantic search

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "legal_contracts"

# ── Chunking (LangChain RecursiveCharacterTextSplitter) ────────────────────────
CHUNK_SIZE      = 600   # characters  (sweet spot for legal clauses)
CHUNK_OVERLAP   = 100   # characters

# ── LLM — Groq ────────────────────────────────────────────────────────────────
# Free tier limits:  8b-instant → 500K tokens/day  |  70b → 100K tokens/day
# We use 8b for both answer generation and judging to stay well within limits.
# For 60 Q&A pairs × 2 modes = 120 answer calls + 120 judge calls ≈ 240K tokens.
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL       = "llama-3.1-8b-instant"   # answer generation (500K/day free)
GROQ_JUDGE_MODEL = "llama-3.1-8b-instant"   # judge  (max_tokens=10, negligible)

# ── Agentic loop ───────────────────────────────────────────────────────────────
MAX_RETRIES          = 3
CONFIDENCE_THRESHOLD = 0.50  # below this → trigger query rewrite
RETRIEVAL_TOP_K      = 5
MIN_RETRIEVAL_SCORE  = 0.55  # top chunk below this → retrieval is genuinely poor

# ── Evaluation ────────────────────────────────────────────────────────────────
RECALL_K = 5   # Recall@5