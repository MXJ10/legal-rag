"""
evaluation/eval.py
──────────────────
Day 5 — Evaluation harness.

Metrics:
  1. Retrieval Recall@K — is the relevant clause in the top-K chunks?
  2. Answer Accuracy    — LLM-as-judge rates answer vs ground truth (0/1)
  3. Rewrite rate       — how often agentic loop needed to rewrite
  4. Confidence delta   — average confidence gain from rewrites

Run: python evaluation/eval.py [--mode baseline|agentic|both]
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR, RECALL_K, GROQ_API_KEY, GROQ_JUDGE_MODEL


def load_qa_pairs(path: Path = None) -> list[dict]:
    path = path or (EVAL_DIR / "qa_pairs.json")
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_contains_answer(chunks: list[dict], qa: dict) -> bool:
    """
    Check if any of the retrieved chunks is from the correct source and clause.
    Heuristic: chunk source matches + clause keyword appears in chunk text.
    """
    source = qa.get("source", "").lower()
    clause = qa.get("relevant_clause", "").lower()
    answer_keywords = qa.get("ground_truth", qa.get("answer", "")).lower().split()[:5]  # first 5 words

    for chunk in chunks:
        chunk_text = chunk["text"].lower()
        chunk_source = chunk.get("source", "").lower()

        # Source match
        source_match = source in chunk_source or chunk_source in source

        # Clause keyword match (any word from clause name appears in chunk)
        clause_match = any(kw in chunk_text for kw in clause.split() if len(kw) > 3)

        # Answer content match (a few key words from expected answer)
        answer_match = sum(1 for kw in answer_keywords if kw in chunk_text) >= 2

        if source_match and (clause_match or answer_match):
            return True

    return False


ACCURACY_JUDGE_PROMPT = """You are evaluating whether a generated answer is correct compared to a ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated}

Is the generated answer correct? It doesn't need to be word-for-word identical, but it must 
capture the key facts without contradicting the ground truth.

Respond with ONLY one of: CORRECT, PARTIAL, INCORRECT"""


def judge_answer_accuracy(question: str, ground_truth: str, generated: str) -> str:
    """
    Use LLM-as-judge to rate answer accuracy.
    Returns: "CORRECT", "PARTIAL", or "INCORRECT"
    Falls back to keyword matching if no LLM available.
    """
    if not GROQ_API_KEY:
        # Fallback: simple keyword overlap
        gt_words = set(ground_truth.lower().split())
        gen_words = set(generated.lower().split())
        overlap = len(gt_words & gen_words) / max(len(gt_words), 1)
        if overlap > 0.4:
            return "CORRECT"
        elif overlap > 0.15:
            return "PARTIAL"
        return "INCORRECT"

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = ACCURACY_JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        generated=generated,
    )
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=GROQ_JUDGE_MODEL,   # small 8B model — uses far fewer tokens
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            verdict = resp.choices[0].message.content.strip().upper()
            return verdict if verdict in ("CORRECT", "PARTIAL", "INCORRECT") else "INCORRECT"
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg and attempt < 3:
                wait = 60 * (attempt + 1)
                print(f"    ⏳ Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ⚠ Judge error: {e}")
                return "INCORRECT"
    return "INCORRECT"


def run_evaluation(mode: str = "both", qa_subset: list[dict] = None) -> dict:
    """
    Run full evaluation. mode = "baseline" | "agentic" | "both"
    Returns metrics dict with per-question results and aggregates.
    """
    from pipeline.retriever import answer_question
    from pipeline.agentic import agentic_answer

    qa_pairs = qa_subset or load_qa_pairs()
    print(f"\n{'='*70}")
    print(f"Running evaluation: {mode.upper()} | {len(qa_pairs)} questions")
    print(f"{'='*70}")

    results = {"baseline": [], "agentic": []}
    modes_to_run = ["baseline", "agentic"] if mode == "both" else [mode]

    for m in modes_to_run:
        print(f"\n--- {m.upper()} ---")
        for i, qa in enumerate(qa_pairs):
            print(f"  [{i+1:2d}/{len(qa_pairs)}] {qa['question'][:60]}...", end=" ")

            if m == "baseline":
                result = answer_question(qa["question"], top_k=RECALL_K, verbose=False)
            else:
                result = agentic_answer(qa["question"], top_k=RECALL_K, verbose=False)

            # Recall@K
            hit = chunk_contains_answer(result["chunks"], qa)

            # Answer accuracy
            accuracy = judge_answer_accuracy(
                qa["question"],
                qa.get("ground_truth", qa.get("answer", "")),
                result["answer"],
            )

            row = {
                "id": qa["id"],
                "question": qa["question"],
                "contract_type": qa["contract_type"],
                "difficulty": qa["difficulty"],
                "question_type": qa.get("question_type", "factual"),
                "recall_hit": hit,
                "accuracy": accuracy,
                "confidence": result.get("confidence", 0),
                "retrieval_method": result.get("retrieval_method", m),
                "num_retries": result.get("num_retries", 0),
                "rewrites_used": result.get("rewrites_used", []),
                "generated_answer": result["answer"],
                "ground_truth": qa.get("ground_truth", qa.get("answer", "")),
            }
            results[m].append(row)
            
            tag = "✓" if hit else "✗"
            acc_tag = {"CORRECT": "✓", "PARTIAL": "~", "INCORRECT": "✗"}.get(accuracy, "?")
            print(f"recall={tag}  accuracy={acc_tag}")

            time.sleep(0.1)   # avoid rate limiting

    # ── Compute aggregates ─────────────────────────────────────────────────────
    summary = {}
    for m in modes_to_run:
        rows = results[m]
        if not rows:
            continue
        n = len(rows)
        recall_at_k = sum(1 for r in rows if r["recall_hit"]) / n
        correct = sum(1 for r in rows if r["accuracy"] == "CORRECT") / n
        partial = sum(1 for r in rows if r["accuracy"] in ("CORRECT", "PARTIAL")) / n
        avg_conf = sum(r["confidence"] for r in rows) / n
        avg_retries = sum(r["num_retries"] for r in rows) / n

        # By difficulty
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in rows if r["difficulty"] == diff]
            if subset:
                diff_acc = sum(1 for r in subset if r["accuracy"] == "CORRECT") / len(subset)
                print(f"  {m} {diff}: {diff_acc:.1%} accuracy ({len(subset)} questions)")

        summary[m] = {
            f"recall_at_{RECALL_K}": round(recall_at_k, 4),
            "accuracy_exact": round(correct, 4),
            "accuracy_partial": round(partial, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_retries": round(avg_retries, 4),
            "n": n,
        }

    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Baseline':>12} {'Agentic':>12} {'Delta':>10}")
    print("-"*70)
    for metric in [f"recall_at_{RECALL_K}", "accuracy_exact", "accuracy_partial", "avg_confidence"]:
        b = summary.get("baseline", {}).get(metric, 0)
        a = summary.get("agentic", {}).get(metric, 0)
        delta = a - b
        label = metric.replace("_", " ").title()
        print(f"{label:<30} {b:>12.1%} {a:>12.1%} {delta:>+10.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"eval_results_{timestamp}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "mode": mode,
        "summary": summary,
        "details": results,
    }, indent=2))
    print(f"\n✅ Results saved to: {out_path}")

    return {"summary": summary, "details": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "agentic", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N questions for quick test")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions before running")
    parser.add_argument("--qa-file", type=Path, default=None, help="Path to QA pairs JSON file")
    args = parser.parse_args()

    qa = load_qa_pairs(args.qa_file)
    if args.offset or args.limit:
        start = args.offset or 0
        end = (start + args.limit) if args.limit else None
        qa = qa[start:end]

    run_evaluation(mode=args.mode, qa_subset=qa)