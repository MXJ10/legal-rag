# Legal Contract Q&A — Agentic RAG

**MIS285N Final Project · UT Austin MSBA**

A Retrieval-Augmented Generation (RAG) system for answering natural-language questions over legal contracts. Compares a **Baseline RAG** pipeline against an **Agentic RAG** loop that rewrites queries when retrieval confidence is low.

---

## Project Structure

```
legal-rag/
├── config.py                  # All hyperparameters and paths
├── pipeline/
│   ├── ingest.py              # PDF/DOCX → text, contract type inference
│   ├── embed.py               # Chunk + embed into ChromaDB
│   ├── retriever.py           # Retrieval, re-ranking, LLM answer generation
│   └── agentic.py             # Agentic query-rewriting loop (3 strategies)
├── evaluation/
│   ├── eval.py                # Evaluation harness (Recall@5, LLM-as-judge)
│   ├── qa_pairs_v3.json       # 60 curated QA pairs (Group A/B/C)
│   └── eval_results_*.json    # Saved eval outputs
├── ui/
│   └── app.py                 # Streamlit Q&A interface
├── data/
│   └── raw/                   # 103 CUAD contract .txt files
└── vectordb/                  # ChromaDB index (not in git — rebuild locally)
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MXJ10/legal-rag.git
cd legal-rag
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
export GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

To make this permanent:
```bash
echo 'export GROQ_API_KEY=gsk_xxxxxxxxxxxx' >> ~/.zshrc
source ~/.zshrc
```

### 5. Build the vector database
`vectordb/` is excluded from git (91 MB). Rebuild it locally:
```bash
python pipeline/embed.py
```
Downloads the embedding model (~80 MB) and indexes all 103 contracts. Takes ~2–3 minutes.

---

## Running the App

```bash
streamlit run ui/app.py
```

Open the URL shown in the terminal. Use the sidebar to switch between **Baseline RAG** and **Agentic RAG** modes, adjust Top-K, and filter by contract type.

---

## Running the Evaluation

```bash
# Full 60-question eval (both modes, ~15 min)
python evaluation/eval.py --qa-file evaluation/qa_pairs_v3.json --mode both

# Quick sanity check (first 10 questions, agentic only)
python evaluation/eval.py --qa-file evaluation/qa_pairs_v3.json --mode agentic --limit 10

# Run a specific slice (questions 21–40)
python evaluation/eval.py --qa-file evaluation/qa_pairs_v3.json --mode both --offset 20 --limit 20
```

Results are saved to `evaluation/eval_results_<timestamp>.json`.

---

## QA Pairs Design

`qa_pairs_v3.json` contains 60 questions across 3 groups:

| Group | Count | Design intent |
|---|---|---|
| **A — Baseline wins** | 29 | Precise legal terminology, verbatim match expected |
| **B — Agentic wins** | 22 | Informal/vague language requiring query rewriting |
| **C — Hard (neither)** | 9 | Multi-clause synthesis, conditional reasoning |

Questions span 10 contracts from the [CUAD dataset](https://www.atticusprojectai.org/cuad).

---

## Key Design Decisions

- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, ~80 MB, fast)
- **Vector DB:** ChromaDB (cosine similarity)
- **Chunk size:** 600 characters, 100-character overlap
- **LLM:** Groq `llama-3.1-8b-instant` (free tier: 500K tokens/day)
- **Agentic strategies:** Legal paraphrase → Decomposition → Disambiguation
- **Rewrite trigger:** Poor retrieval confidence (<0.50 + top chunk <0.55) OR informal signal words in the question
- **Confidence scoring:** Blended (70% LLM-rated + 30% cosine heuristic)
- **Metadata re-ranking:** Boosts chunks from contracts named in the question (+0.15)
