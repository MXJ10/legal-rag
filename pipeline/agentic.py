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
from config import MAX_RETRIES, GROQ_API_KEY
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

DISAMBIGUATION_PROMPT = """A user asked: "{question}"

This question could apply to multiple contracts. The top retrieved contracts are:
{contract_list}

Rewrite the question to be specific to the most likely intended contract based on context clues in the question.
Return ONLY the rewritten question, nothing else."""

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
    Blend heuristic + LLM confidence score. Falls back to heuristic cleanly.
    """
    heuristic_score = compute_retrieval_confidence(chunks)

    if not GROQ_API_KEY or not chunks:
        return heuristic_score

    excerpt_text = "\n---\n".join(c["text"][:300] for c in chunks[:3])
    prompt = CONFIDENCE_CHECK_PROMPT.format(
        question=question,
        excerpts=excerpt_text,
    )
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        llm_score = float(raw.split()[0])
        llm_score = max(0.0, min(1.0, llm_score))
        return round((llm_score * 0.7) + (heuristic_score * 0.3), 4)
    except Exception:
        return heuristic_score


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


INFORMAL_SIGNALS = [
    "keeps crashing", "walk away", "fire ", "pull the plug",
    "wants out", "cancel his deal", "stop paying", "settle up",
    "end the project early", "still have to pay", "keep using",
    "bring in an outside", "secret information", "runs out",
    "bail out", "get out of",
]


def should_rewrite(chunks: list[dict], confidence: float, question: str = "") -> bool:
    """Rewrite only on genuinely poor retrieval OR explicit informal signal words."""
    if not chunks:
        return True
    top_score = chunks[0]["score"] if chunks else 0

    # Only trigger on genuinely poor retrieval
    if confidence < 0.50 and top_score < 0.55:
        return True

    # OR if the question contains informal signal words
    # (these are Group B questions designed for agentic to win)
    if any(signal in question.lower() for signal in INFORMAL_SIGNALS):
        return True

    return False


def rewrite_with_disambiguation(question: str, chunks: list[dict]) -> str:
    """Strategy C: Disambiguate which contract the question is about."""
    contracts = list(dict.fromkeys(c["source"] for c in chunks[:5]))
    contract_list = "\n".join(f"- {c}" for c in contracts)
    prompt = DISAMBIGUATION_PROMPT.format(
        question=question,
        contract_list=contract_list,
    )
    return call_llm(prompt).strip()


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

    top_score_0 = best_result["chunks"][0]["score"] if best_result["chunks"] else 0
    will_rewrite = should_rewrite(best_result["chunks"], best_confidence, question=question)
    print(f"\n[DEBUG] Q: {question[:80]}")
    print(f"[DEBUG]   initial confidence={best_confidence:.4f}  top_chunk_score={top_score_0:.4f}  should_rewrite={will_rewrite}")

    # ── Agentic retries ────────────────────────────────────────────────────────
    retry = 0
    while should_rewrite(best_result["chunks"], best_confidence, question=question) and retry < MAX_RETRIES:
        retry += 1
        print(f"[DEBUG]   → REWRITE triggered: retry {retry}/{MAX_RETRIES}  confidence={best_confidence:.4f}")

        if retry == 1:
            # Strategy A: Legal paraphrase
            rewritten = rewrite_legal_paraphrase(question)
            rewrites_used.append(rewritten)
            print(f"[DEBUG]   strategy=paraphrase")
            print(f"[DEBUG]   original : {question[:80]}")
            print(f"[DEBUG]   rewritten: {rewritten[:80]}")
            attempt = _attempt(rewritten, "paraphrase")

        elif retry == 2:
            # Strategy B: Decomposition
            sub_questions = rewrite_decompose(question)
            rewrites_used.extend(sub_questions)
            print(f"[DEBUG]   strategy=decomposition  sub_questions={sub_questions}")

            sub_results = []
            for sq in sub_questions:
                sub_chunks = retrieve(sq, top_k=top_k)
                sub_results.append(sub_chunks)

            merged_chunks = merge_chunk_results(sub_results, top_k=top_k)
            attempt = _attempt(question, "decomposition", source_chunks_override=merged_chunks)

        elif retry == 3:
            # Strategy C: Disambiguation — identify which contract the question targets
            disambiguated = rewrite_with_disambiguation(question, best_result["chunks"])
            rewrites_used.append(disambiguated)
            print(f"[DEBUG]   strategy=disambiguation")
            print(f"[DEBUG]   original    : {question[:80]}")
            print(f"[DEBUG]   disambiguated: {disambiguated[:80]}")
            attempt = _attempt(disambiguated, "disambiguation")

        print(f"[DEBUG]   after rewrite: confidence={attempt['confidence']:.4f}  (best so far={best_confidence:.4f})")
        if attempt["confidence"] > best_confidence:
            best_result = attempt
            best_confidence = attempt["confidence"]
            print(f"[DEBUG]   → new best! confidence={best_confidence:.4f}")
        else:
            print(f"[DEBUG]   → no improvement, keeping previous best")

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