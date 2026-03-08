"""
pipeline/embed.py
─────────────────
Day 1 — Step 3: Load chunks from data/processed/chunks.jsonl,
generate embeddings with all-MiniLM-L6-v2, and index into ChromaDB.

ChromaDB stores:
  - embeddings (384-dim vectors from MiniLM)
  - documents (chunk text)
  - metadata (source, contract_type, chunk_id, etc.)

Run: python pipeline/embed.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROC_DIR, VECTORDB_DIR,
    EMBEDDING_MODEL, CHROMA_COLLECTION
)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "chromadb", "sentence-transformers", "-q"])
    import chromadb
    from sentence_transformers import SentenceTransformer


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client stored at VECTORDB_DIR."""
    return chromadb.PersistentClient(path=str(VECTORDB_DIR))


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get existing collection or create fresh one."""
    try:
        # Delete existing collection to allow re-indexing
        client.delete_collection(CHROMA_COLLECTION)
        print(f"  Deleted existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
    )
    print(f"  Created collection '{CHROMA_COLLECTION}'")
    return collection


def embed_and_index(batch_size: int = 64):
    chunks_path = DATA_PROC_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"⚠ No chunks found at {chunks_path}")
        print("  Run: python pipeline/ingest.py first")
        return

    chunks = load_chunks(chunks_path)
    print(f"\n{'='*60}")
    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")
    print(f"{'='*60}")

    # Load embedding model
    print("Loading sentence-transformers model (downloads ~80MB on first run)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Model loaded | Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Extract texts
    texts = [c["text"] for c in chunks]

    # Batch encode with progress
    print(f"  Encoding in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit norm → cosine = dot product
    )
    print(f"  ✓ Embeddings shape: {embeddings.shape}")

    # Index into ChromaDB
    print("\nIndexing into ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # ChromaDB batch upsert
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embs.tolist(),
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "source": c["source"],
                "contract_type": c["contract_type"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
            } for c in batch_chunks],
        )
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks", end="\r")

    print(f"\n  ✓ All {len(chunks)} chunks indexed")

    # Quick sanity check: run a test query
    print("\nRunning sanity check query...")
    test_query = "What is the confidentiality obligation?"
    test_emb = model.encode([test_query], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=test_emb.tolist(),
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    print(f"  Query: '{test_query}'")
    for j, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist  # cosine distance → similarity
        print(f"  [{j+1}] score={score:.3f} | {meta['source']} | {doc[:80]}...")

    print(f"\n{'='*60}")
    print(f"✅ ChromaDB index ready at: {VECTORDB_DIR}")
    print(f"   Collection : {CHROMA_COLLECTION}")
    print(f"   Documents  : {collection.count()}")
    print(f"{'='*60}")
    print("\nNext step: python pipeline/retriever.py  (Day 2)")


if __name__ == "__main__":
    embed_and_index()