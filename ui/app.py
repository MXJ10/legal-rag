"""
ui/app.py
─────────
Day 6 — Streamlit UI for Legal Contract Q&A

Run: streamlit run ui/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

from config import GROQ_API_KEY, CONFIDENCE_THRESHOLD

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Contract Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Legal Contract Q&A")
st.caption("MIS285N — Agentic RAG for Legal Documents | UT Austin")

# ── Sidebar config ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Retrieval Mode",
        ["Baseline RAG", "Agentic RAG"],
        index=1,
        help="Agentic RAG rewrites queries when confidence is low",
    )
    
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)
    
    contract_type = st.selectbox(
        "Filter by contract type",
        ["All", "nda", "lease", "employment", "development", "marketing",
         "consulting", "ip_license", "manufacturing", "outsourcing",
         "transportation", "service"],
        help="Filter retrieval to a specific contract category (Baseline only)",
    )
    
    st.divider()
    st.header("📊 Session Stats")
    if "stats" not in st.session_state:
        st.session_state.stats = {"questions": 0, "rewrites": 0, "avg_conf": []}

    col1, col2 = st.columns(2)
    col1.metric("Questions", st.session_state.stats["questions"])
    col2.metric("Rewrites", st.session_state.stats["rewrites"])
    
    if st.session_state.stats["avg_conf"]:
        avg = sum(st.session_state.stats["avg_conf"]) / len(st.session_state.stats["avg_conf"])
        st.metric("Avg Confidence", f"{avg:.2f}")
    
    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not set.\nSet it to enable real LLM answers.\n\nFree key: console.groq.com")

# ── Main Q&A interface ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# Example questions
with st.expander("💡 Example questions", expanded=False):
    examples = [
        "How long do I have to reject the developer's work?",
        "Can I do the same work for a competitor?",
        "What counts as confidential information under this agreement?",
        "What happens if we can't agree on a statement of work?",
        "Will this agreement renew automatically?",
        "Who is responsible for taxes — me or the company?",
        "Can they use my name or logo without asking me?",
        "How much notice do I need to give to terminate the agreement?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.current_question = ex

question = st.text_input(
    "Ask a question about your contracts:",
    value=st.session_state.get("current_question", ""),
    placeholder="e.g. What are the termination conditions?",
    key="question_input",
)

col_ask, col_clear = st.columns([1, 5])
ask_btn = col_ask.button("Ask ⚖️", type="primary", use_container_width=True)
col_clear.button("Clear history", on_click=lambda: st.session_state.history.clear())

if ask_btn and question:
    spinner_msg = "Retrieving relevant clauses..." if mode == "Baseline RAG" else "Running agentic retrieval (may rewrite query up to 3×)..."
    with st.spinner(spinner_msg):
        filter_type = None if contract_type == "All" else contract_type

        if mode == "Baseline RAG":
            from pipeline.retriever import answer_question
            result = answer_question(question, top_k=top_k, contract_type_filter=filter_type)
        else:
            from pipeline.agentic import agentic_answer
            result = agentic_answer(question, top_k=top_k)

    # Update stats
    st.session_state.stats["questions"] += 1
    st.session_state.stats["rewrites"] += result.get("num_retries", 0)
    st.session_state.stats["avg_conf"].append(result.get("confidence", 0))

    # Store in history
    st.session_state.history.insert(0, {"question": question, "result": result, "mode": mode})

# ── Display history ────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    q = entry["question"]
    r = entry["result"]
    m = entry["mode"]

    with st.container():
        st.markdown(f"**Q: {q}**")
        
        # Confidence + method badges
        conf = r.get("confidence", 0)
        conf_color = "green" if conf > 0.6 else "orange" if conf > 0.3 else "red"
        method = r.get("retrieval_method", "baseline")
        retries = r.get("num_retries", 0)

        badge_cols = st.columns([1, 1, 1, 4])
        badge_cols[0].markdown(f":{conf_color}[Confidence: {conf:.2f}]")
        badge_cols[1].markdown(f":blue[Method: {method}]")
        if retries > 0:
            badge_cols[2].markdown(f":orange[Rewrites: {retries}]")

        st.markdown(r.get("answer", "No answer generated."))

        # Show rewrites if any
        if r.get("rewrites_used"):
            with st.expander(f"🔄 Query rewrites ({len(r['rewrites_used'])})"):
                for rw in r["rewrites_used"]:
                    st.markdown(f"- _{rw}_")

        # Show retrieved chunks
        with st.expander(f"📄 Retrieved chunks (top {len(r.get('chunks', []))})"):
            for i, chunk in enumerate(r.get("chunks", [])[:5], 1):
                st.markdown(
                    f"**[{i}]** `{chunk['source']}` · `{chunk['contract_type']}` "
                    f"· score `{chunk['score']:.3f}`"
                )
                st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
                st.divider()

        st.divider()