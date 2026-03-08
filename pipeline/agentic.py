"""
pipeline/agentic.py
────────────────────
Day 3 — Agentic query-rewriting loop.

Flow:
  1. Retrieve with original query
  2. Compute confidence score
  3. If confidence < threshold → try rewrite strategies:
       Strategy A: LLM paraphrases using legal terminology
       Strategy B: LLM decomposes into sub-questions (retrieve each, merge)
  4. Repeat up to MAX_RETRIES
  5. Return best result across all attempts

This file is scaffolded on Day 1 so the team can see the full architecture.
Complete implementation happens on Day 3.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_RETRIES, CONFIDENCE_THRESHOLD, GROQ_API_KEY
from pipeline.retriever import retrieve, compute_retrieval_confidence, call_llm, build_prompt


# ── Rewrite prompts ────────────────────────────────────────────────────────────

LEGAL_PARAPHRASE_PROMPT = """You are a legal expert. Rewrite the following question using precise legal terminology 
as it would appear in a contract. Keep the meaning identical but use formal legal language.

Original question: {question}

Rewritten question (one sentence, legal terminology, no explanation):"""

DECOMPOSE_PROMPT = """You are a legal analyst. Decompose the following complex question into 2-3 simpler 
sub-questions that together answer the original. Each sub-question should target a specific 
clause type found in legal contracts.

Original question: {question}

Output format (return ONLY this, no extra text):
SUB1: <first sub-question>
SUB2: <second sub-question>
SUB3: <third sub-question or NONE>"""

CONFIDENCE_CHECK_PROMPT = """You are evaluating whether retrieved contract excerpts are sufficient to answer a question.

Question: {question}

Retrieved excerpts:
{excerpts}

Rate your confidence that these excerpts contain enough information to accurately answer the question.
Respond with ONLY a number between 0.0 and 1.0, where:
  0.0 = excerpts are completely irrelevant
  0.5 = excerpts are partially relevant
  1.0 = excerpts fully answer the question

Confidence score (number only):"""


def llm_confidence_check(question: str, chunks: list[dict]) -> float:
    """
    Ask the LLM to rate retrieval quality. More accurate than cosine heuristic.
    Falls back to heuristic if LLM unavailable.
    """
    if not GROQ_API_KEY:
        return compute_retrieval_confidence(chunks)

    excerpt_text = "\n---\n".join(c["text"][:200] for c in chunks[:3])
    prompt = CONFIDENCE_CHECK_PROMPT.format(
        question=question,
        excerpts=excerpt_text,
    )
    
    try:
        response = call_llm(prompt)
        score = float(response.strip().split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        return compute_retrieval_confidence(chunks)


def rewrite_legal_paraphrase(question: str) -> str:
    """Strategy A: Rephrase using legal terminology."""
    prompt = LEGAL_PARAPHRASE_PROMPT.format(question=question)
    rewritten = call_llm(prompt)
    # Strip any preamble the model might add
    for prefix in ["Rewritten question:", "Answer:", "Legal question:"]:
        if rewritten.lower().startswith(prefix.lower()):
            rewritten = rewritten[len(prefix):].strip()
    return rewritten.strip()


def rewrite_decompose(question: str) -> list[str]:
    """Strategy B: Decompose into sub-questions."""
    prompt = DECOMPOSE_PROMPT.format(question=question)
    response = call_llm(prompt)
    
    sub_questions = []
    for line in response.strip().splitlines():
        for prefix in ["SUB1:", "SUB2:", "SUB3:"]:
            if line.upper().startswith(prefix):
                q = line[len(prefix):].strip()
                if q and q.upper() != "NONE":
                    sub_questions.append(q)
    
    return sub_questions if sub_questions else [question]


def merge_chunk_results(results_list: list[list[dict]], top_k: int = 5) -> list[dict]:
    """
    Merge and deduplicate chunks from multiple retrieval rounds.
    Ranks by score, deduplicates by chunk_id (using text as proxy).
    """
    seen_texts = set()
    merged = []
    for chunks in results_list:
        for chunk in chunks:
            key = chunk["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(chunk)
    
    merged.sort(key=lambda c: c["score"], reverse=True)
    return merged[:top_k]


def agentic_answer(
    question: str,
    top_k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Main agentic loop. Returns the best answer found within MAX_RETRIES.
    
    Returns:
      {
        question, answer, chunks, confidence,
        retrieval_method,  # "baseline" | "paraphrase" | "decomposition"
        rewrites_used,     # list of rewritten queries tried
        num_retries,
        all_attempts,      # list of {query, confidence, method}
      }
    """
    all_attempts = []
    rewrites_used = []
    best_result = None
    best_confidence = -1.0

    def _attempt(query: str, method: str, source_chunks_override=None) -> dict:
        chunks = source_chunks_override or retrieve(query, top_k=top_k)
        confidence = llm_confidence_check(question, chunks)   # always judge against ORIGINAL question
        attempt = {
            "query": query,
            "method": method,
            "confidence": confidence,
            "chunks": chunks,
        }
        all_attempts.append(attempt)
        if verbose:
            print(f"  [{method}] confidence={confidence:.3f} | query: {query[:70]}")
        return attempt

    # ── Attempt 0: Baseline ────────────────────────────────────────────────────
    attempt = _attempt(question, "baseline")
    if attempt["confidence"] > best_confidence:
        best_result = attempt
        best_confidence = attempt["confidence"]

    # ── Agentic retries ────────────────────────────────────────────────────────
    retry = 0
    while best_confidence < CONFIDENCE_THRESHOLD and retry < MAX_RETRIES:
        retry += 1
        if verbose:
            print(f"\n  → Low confidence ({best_confidence:.3f}), retry {retry}/{MAX_RETRIES}")

        if retry == 1 or retry == 3:
            # Strategy A: Legal paraphrase
            rewritten = rewrite_legal_paraphrase(question)
            rewrites_used.append(rewritten)
            attempt = _attempt(rewritten, "paraphrase")

        elif retry == 2:
            # Strategy B: Decomposition
            sub_questions = rewrite_decompose(question)
            rewrites_used.extend(sub_questions)
            if verbose:
                print(f"  Decomposed into {len(sub_questions)} sub-questions")
            
            sub_results = []
            for sq in sub_questions:
                sub_chunks = retrieve(sq, top_k=top_k)
                sub_results.append(sub_chunks)
            
            merged_chunks = merge_chunk_results(sub_results, top_k=top_k)
            attempt = _attempt(question, "decomposition", source_chunks_override=merged_chunks)

        if attempt["confidence"] > best_confidence:
            best_result = attempt
            best_confidence = attempt["confidence"]

    # ── Generate final answer with best chunks ─────────────────────────────────
    best_chunks = best_result["chunks"]
    prompt = build_prompt(question, best_chunks)
    answer = call_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "chunks": best_chunks,
        "confidence": best_confidence,
        "retrieval_method": best_result["method"],
        "rewrites_used": rewrites_used,
        "num_retries": retry,
        "all_attempts": [
            {"query": a["query"], "method": a["method"], "confidence": a["confidence"]}
            for a in all_attempts
        ],
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Legal Contract Q&A — Agentic RAG")
    print("="*60)
    
    test_questions = [
        "What happens if I want to leave early?",          # vague — lease/employment ambiguous
        "Can I tell my friends about this deal?",          # very informal — NDA question
        "What are my obligations?",                        # extremely vague
        "How much notice do I need to give?",              # multi-contract ambiguity
    ]
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        result = agentic_answer(q, verbose=True)
        print(f"Best method : {result['retrieval_method']}")
        print(f"Retries used: {result['num_retries']}")
        print(f"Final conf  : {result['confidence']:.3f}")
        if result['rewrites_used']:
            print(f"Rewrites    : {result['rewrites_used']}")
        print(f"Answer      : {result['answer'][:200]}...")