"""
pipeline/retriever.py
─────────────────────
Day 2 (stub ready on Day 1) — Baseline RAG retriever.

This is the NAIVE baseline: embed query → retrieve top-K chunks → pass to LLM.
No query rewriting. No agentic loop. That comes in agentic.py (Day 3).

Usage (after ingest + embed):
  python pipeline/retriever.py                         # interactive demo
  from pipeline.retriever import retrieve, answer_question
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VECTORDB_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL,
    RETRIEVAL_TOP_K, GROQ_API_KEY, GROQ_MODEL
)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "chromadb", "sentence-transformers", "-q"])
    import chromadb
    from sentence_transformers import SentenceTransformer

# ── Module-level singletons (loaded once) ─────────────────────────────────────
_model: Optional[SentenceTransformer] = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
        _collection = client.get_collection(CHROMA_COLLECTION)
    return _collection


# ── Core retrieval function ────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    contract_type_filter: Optional[str] = None,
) -> list[dict]:
    """
    Embed query and retrieve top-K chunks from ChromaDB.
    
    Returns list of dicts:
      {text, source, contract_type, score, chunk_index}
    """
    model = _get_model()
    collection = _get_collection()

    query_emb = model.encode([query], normalize_embeddings=True)

    where_filter = None
    if contract_type_filter:
        where_filter = {"contract_type": {"$eq": contract_type_filter}}

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "contract_type": meta.get("contract_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "score": round(1 - dist, 4),    # cosine similarity (higher = better)
            "distance": round(dist, 4),
        })

    return chunks


def compute_retrieval_confidence(chunks: list[dict]) -> float:
    """
    Heuristic confidence score for retrieved chunks.
    Used by the agentic loop to decide whether to rewrite query.
    
    Score = average of top-3 similarities.
    Range: 0.0 (terrible retrieval) → 1.0 (perfect match)
    """
    if not chunks:
        return 0.0
    top3_scores = sorted([c["score"] for c in chunks], reverse=True)[:3]
    return round(sum(top3_scores) / len(top3_scores), 4)


# ── LLM answer generation ──────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} from '{chunk['source']}' ({chunk['contract_type']})]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful legal contract analyst. A user has asked a question about their contracts.
Use the contract excerpts below to answer. Synthesize information across excerpts if needed.

Instructions:
1. Write a clear, direct answer in 2-4 sentences.
2. If the exact answer is not stated but can be reasonably inferred from the excerpts, say so.
3. End with "Source: [Excerpt X from 'filename']" citing the most relevant excerpt(s).
4. Only say "Not found in provided excerpts" if the topic is completely absent from all excerpts.

CONTRACT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def call_llm(prompt: str) -> str:
    """
    Call Groq llama-3.1-8b-instant for answer generation.
    Free tier: 500K tokens/day — sufficient for full eval runs (≈240K tokens).

    To enable: export GROQ_API_KEY=your_key_here
    Get a free key at: https://console.groq.com (no credit card)
    """
    if not GROQ_API_KEY:
        return (
            "[LLM stub — set GROQ_API_KEY to enable real answers]\n"
            f"(Would answer using {len(prompt.split())} word prompt)"
        )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def answer_question(
    question: str,
    top_k: int = RETRIEVAL_TOP_K,
    contract_type_filter: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    End-to-end baseline RAG: retrieve + answer.
    
    Returns:
      {question, answer, chunks, confidence, retrieval_method}
    """
    chunks = retrieve(question, top_k=top_k, contract_type_filter=contract_type_filter)
    confidence = compute_retrieval_confidence(chunks)
    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)

    if verbose:
        print(f"\nQ: {question}")
        print(f"Retrieval confidence: {confidence:.3f}")
        print(f"Top chunk score: {chunks[0]['score'] if chunks else 'N/A'}")
        print(f"A: {answer}\n")

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "confidence": confidence,
        "retrieval_method": "baseline",
        "num_chunks_retrieved": len(chunks),
    }


# ── Interactive demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Legal Contract Q&A — Baseline Retriever")
    print("="*60)
    print("Type a question about your contracts. Type 'quit' to exit.")
    print("(Set GROQ_API_KEY env var for real LLM answers)\n")

    # Quick smoke test
    sample_questions = [
        "What are the confidentiality obligations?",
        "How long is the lease term?",
        "What is the employee's base salary?",
        "What happens if a party breaches the agreement?",
    ]

    print("Running smoke test with sample questions...\n")
    for q in sample_questions:
        result = answer_question(q, verbose=True)

    # Interactive loop
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            result = answer_question(q, verbose=True)
            print("\nRetrieved chunks:")
            for i, chunk in enumerate(result["chunks"][:3], 1):
                print(f"  [{i}] {chunk['source']} | score={chunk['score']} | {chunk['text'][:100]}...")