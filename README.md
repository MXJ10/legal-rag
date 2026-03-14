# Legal Contract Q&A — Agentic RAG

A Retrieval-Augmented Generation system for answering natural-language questions over legal contracts. Users can ask questions in plain English — formal or informal — and receive grounded answers citing the relevant contract. The system compares a standard baseline RAG pipeline against an agentic loop that automatically rewrites queries using legal paraphrase, decomposition, or disambiguation strategies when retrieval confidence is low.

## Quick Start

```bash
git clone https://github.com/MXJ10/legal-rag.git
cd legal-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get a free API key at [console.groq.com](https://console.groq.com) and set it:

```bash
export GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

Build the vector index (first run only, ~2–3 minutes):

```bash
python pipeline/embed.py
```

Launch the app:

```bash
streamlit run ui/app.py
```

## Tech Stack

ChromaDB · LangChain · Groq (llama-3.1-8b-instant) · Streamlit · sentence-transformers (all-MiniLM-L6-v2)

## Team

Alisha Surabhi · Mihir J Gandham · Shivangi Gupta · Simoni K Dalal

MIS285N — Generative AI | UT Austin MSBA | Spring 2026
