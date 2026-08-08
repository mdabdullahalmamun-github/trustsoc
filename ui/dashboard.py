"""Streamlit dashboard — visual front-end reusing trustsoc_answer() from
pipeline/rag_pipeline.py (no duplicate retrieval logic). Lets the user pick
a model, toggle RAG on/off, adjust how many sources are retrieved, and see
a live trust-signals panel (Grounded / Citations / Hedged) alongside the
answer and its retrieved sources."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pipeline.rag_pipeline import trustsoc_answer

st.set_page_config(page_title="TrustSOC", layout="wide")
st.title("TrustSOC — Trustworthy SOC Assistant")
st.caption("MSc Cybersecurity · Md Abdullah Al Mamun · Student ID: 2010022")

model = st.sidebar.selectbox("Model", ["mistral", "llama3"])
use_rag = st.sidebar.checkbox("Use RAG (retrieval)", value=True)
k = st.sidebar.slider("Sources to retrieve", 1, 10, 5)
# Note: llama3 + k >= 10 has crashed the local Ollama backend on 4GB VRAM
# (see build log, Phase 3.4, Dashboard Test 2) — keep k=5 as the safe default.

q = st.text_input("Analyst query", "What ATT&CK technique is encoded PowerShell execution?")

if st.button("Analyse") and q:
    result = trustsoc_answer(q, model=model, use_rag=use_rag, k=k)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(f"{result['seconds']}s · model: {model} · RAG: {use_rag}")
    with c2:
        st.subheader("Retrieved sources")
        if result["contexts"]:
            for i, c in enumerate(result["contexts"], 1):
                st.markdown(f"**{i}.** {c[:200]}...")
        else:
            st.info("No retrieval (baseline mode).")

    st.divider()
    st.subheader("Trust signals")
    # Simple heuristic hedge detector: checks for phrases the citation-enforcing
    # prompt asks the model to use when it lacks enough context to answer.
    hedges = ["does not contain enough information", "cannot", "insufficient"]
    hedged = any(h in result["answer"].lower() for h in hedges)

    s1, s2, s3 = st.columns(3)
    s1.metric("Grounded?", "Yes" if result["contexts"] else "No (baseline)")
    s2.metric("Citations", result["answer"].count("["))
    s3.metric("Hedged?", "Yes" if hedged else "No")