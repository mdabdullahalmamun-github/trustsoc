"""Streamlit dashboard — visual front-end reusing trustsoc_answer() from
pipeline/rag_pipeline.py (no duplicate retrieval logic). Styled as a SOC
monitoring console rather than a generic chat UI: dark surface, monospace
for technical data (IDs, distances, timings), and colour used only for
functional status (grounded / fabricated / hedged), matching how real SOC
tooling uses colour for alert severity rather than decoration.

Exposes the full pipeline, not just a subset: model choice (with an
optional side-by-side comparison of both models), RAG on/off (with an
optional side-by-side comparison of both settings — combinable with the
model comparison for up to 4 panels at once), k, and the two advanced
parameters added to the retriever — relevance threshold and
source-balancing — so the pipeline's actual tunable behaviour is visible
and adjustable here, not just hardcoded."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pipeline.rag_pipeline import trustsoc_answer, MODEL_A, MODEL_B, RELEVANCE_THRESHOLD

st.set_page_config(page_title="TrustSOC", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Theme: console-style dark UI. Colour is functional (status), not decorative.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0B0F14;
    --panel: #131A21;
    --hairline: #232B35;
    --text: #E6EDF3;
    --muted: #7C8B9A;
    --grounded: #3FB98A;
    --fabricated: #E5484D;
    --hedged: #E8A33D;
    --attack: #5B8DEF;
    --nvd: #E8A33D;
    --kev: #E5484D;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--hairline); }
h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }

.mono { font-family: 'JetBrains Mono', monospace; }

.console-header {
    display: flex; align-items: baseline; gap: 0.75rem;
    padding-bottom: 0.75rem; border-bottom: 1px solid var(--hairline); margin-bottom: 1.25rem;
}
.console-header .brand { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
.console-header .subtitle { color: var(--muted); font-size: 0.9rem; }
.console-header .meta {
    margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--muted);
}

.panel {
    background: var(--panel); border: 1px solid var(--hairline); border-radius: 6px;
    padding: 1rem 1.1rem; margin-bottom: 0.75rem;
}
.panel-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem;
}

.badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.35rem; border: 1px solid transparent;
}
.badge-model { background: rgba(91,141,239,0.12); color: var(--attack); border-color: rgba(91,141,239,0.3); }
.badge-time { background: rgba(124,139,154,0.12); color: var(--muted); border-color: var(--hairline); }

.source-chip {
    display: block; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    padding: 0.5rem 0.6rem; margin-bottom: 0.4rem; border-radius: 5px;
    background: rgba(255,255,255,0.02); border-left: 3px solid var(--hairline);
}
.source-chip .tag { font-weight: 700; margin-right: 0.4rem; }
.source-chip .dist { color: var(--muted); float: right; }
.source-chip .text { color: var(--muted); display: block; margin-top: 0.25rem; font-family: 'Inter', sans-serif; font-size: 0.82rem; }
.src-attack { border-left-color: var(--attack); } .src-attack .tag { color: var(--attack); }
.src-nvd    { border-left-color: var(--nvd); }    .src-nvd .tag    { color: var(--nvd); }
.src-kev    { border-left-color: var(--kev); }    .src-kev .tag    { color: var(--kev); }

.cite-chip {
    display: inline-flex; align-items: center; gap: 0.3rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 0.2rem 0.55rem; border-radius: 4px; margin: 0.15rem 0.35rem 0.15rem 0;
}
.cite-ok  { background: rgba(63,185,138,0.12); color: var(--grounded); border: 1px solid rgba(63,185,138,0.35); }
.cite-bad { background: rgba(229,72,77,0.12); color: var(--fabricated); border: 1px solid rgba(229,72,77,0.35); }

.trust-strip { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.trust-pill {
    flex: 1; min-width: 150px; background: var(--panel); border: 1px solid var(--hairline);
    border-radius: 6px; padding: 0.7rem 0.9rem;
}
.trust-pill .label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted); }
.trust-pill .value { font-size: 1.05rem; font-weight: 600; margin-top: 0.15rem; }
.pill-good .value { color: var(--grounded); }
.pill-bad  .value { color: var(--fabricated); }
.pill-warn .value { color: var(--hedged); }
.pill-neutral .value { color: var(--text); }

.refusal-banner {
    background: rgba(232,163,61,0.08); border: 1px solid rgba(232,163,61,0.35);
    border-radius: 6px; padding: 0.7rem 0.9rem; color: var(--hedged);
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="console-header">
    <span class="brand">TRUSTSOC</span>
    <span class="subtitle">SOC Analyst Assistant — Trustworthy RAG Console</span>
    <span class="meta">Md Abdullah Al Mamun · 2010022 · MSc Cybersecurity</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls — exposes the full pipeline, not a hardcoded subset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="panel-title">Query settings</div>', unsafe_allow_html=True)
    compare_models = st.checkbox("Compare both models", value=False)
    compare_rag = st.checkbox("Compare RAG vs no-RAG", value=False)
    if not compare_models:
        model = st.selectbox("Model", [MODEL_A, MODEL_B])
    if not compare_rag:
        use_rag = st.checkbox("Use RAG (retrieval)", value=True)
    k = st.slider("Sources to retrieve (max)", 1, 10, 5)
    st.caption("This is a ceiling, not a guarantee — fewer sources are returned if fewer pass the relevance threshold.")

    with st.expander("Advanced — retrieval tuning"):
        threshold = st.slider(
            "Relevance threshold (cosine distance)", 0.1, 1.0, RELEVANCE_THRESHOLD, 0.05,
            help="Chunks beyond this distance are discarded rather than handed to the model. Lower = stricter.",
        )
        balance = st.checkbox(
            "Source-balanced retrieval", value=True,
            help="Guarantees roughly half the results come from MITRE ATT&CK, half from NVD/KEV, so the much larger CVE volume doesn't drown out technique matches.",
        )

    st.markdown("---")
    st.caption("Known hardware limit: Models specifically llama3 at sources to retrieve value k≥10 has intermittently crashed the local Ollama backend (4GB VRAM) under memory pressure — not guaranteed every run, but k=5 is the reliable default.")

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
q = st.text_input("Analyst query", "What ATT&CK technique is encoded PowerShell execution?")
run = st.button("Analyse", type="primary")

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
SOURCE_CLASS = {"MITRE ATT&CK": "src-attack", "NVD": "src-nvd", "CISA KEV": "src-kev"}
SOURCE_TAG = {"MITRE ATT&CK": "ATT&CK", "NVD": "NVD", "CISA KEV": "KEV"}

def render_answer_panel(result):
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Answer</div>
        <span class="badge badge-model">{result['model']}</span>
        <span class="badge badge-time">{result['seconds']}s</span>
        <span class="badge badge-time">{'RAG' if result['use_rag'] else 'no retrieval'}</span>
        <p style="margin-top:0.75rem; line-height:1.5;">{result['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

    if result["use_rag"] and not result["contexts"]:
        st.markdown(f"""
        <div class="refusal-banner">⚠ No chunks passed the relevance threshold — the model was not
        called; the system returned the standard refusal directly rather than guessing.</div>
        """, unsafe_allow_html=True)

def render_sources_panel(result):
    st.markdown('<div class="panel"><div class="panel-title">Retrieved sources</div>', unsafe_allow_html=True)
    if not result["contexts"]:
        st.markdown('<span class="mono" style="color:var(--muted);">— none —</span></div>', unsafe_allow_html=True)
        return
    chips = ""
    for hit in result.get("hits", []):
        cls = SOURCE_CLASS.get(hit["source"], "")
        tag = SOURCE_TAG.get(hit["source"], hit["source"] or "?")
        chips += f"""<div class="source-chip {cls}">
            <span class="tag">{tag}</span><span class="mono">{hit['id']}</span>
            <span class="dist mono">d={hit['distance']}</span>
            <span class="text">{hit['text'][:120]}...</span>
        </div>"""
    st.markdown(chips + "</div>", unsafe_allow_html=True)

def render_citation_ledger(result):
    cc = result.get("citation_check")
    st.markdown('<div class="panel"><div class="panel-title">Citation ledger</div>', unsafe_allow_html=True)
    if not cc or not cc["cited"]:
        st.markdown('<span class="mono" style="color:var(--muted);">no IDs cited in this answer</span></div>', unsafe_allow_html=True)
        return
    chips = ""
    for cid in cc["cited"]:
        ok = cid in cc["grounded"]
        cls = "cite-ok" if ok else "cite-bad"
        icon = "✓" if ok else "✗"
        chips += f'<span class="cite-chip {cls}">{icon} {cid}</span>'
    st.markdown(chips + "</div>", unsafe_allow_html=True)

def render_trust_strip(result):
    grounded = bool(result["contexts"])
    cc = result.get("citation_check")
    fabricated = bool(cc and cc["has_fabrication"])
    # Match on the stable opening fragment of the refusal message rather than
    # the full sentence — models often paraphrase the ending (e.g. "...to
    # determine X" instead of "...to answer this reliably"), so an exact
    # full-sentence match misses real hedges. This fragment is specific
    # enough to the prompt's own wording (Rule 3) not to false-positive on
    # unrelated text, while being robust to that paraphrasing.
    hedged = "does not contain enough information" in result["answer"].lower()
    n_cited = len(cc["cited"]) if cc else 0

    if not result["use_rag"]:
        # Citation validation only means something when there's retrieved
        # context to check IDs against — with RAG off, any IDs the model
        # mentions are unvalidated by construction, not "zero citations".
        citations_value = "N/A (no retrieval)"
        citations_class = "pill-neutral"
    elif fabricated:
        citations_value = f"{n_cited} cited · FABRICATED"
        citations_class = "pill-bad"
    elif n_cited:
        citations_value = f"{n_cited} cited"
        citations_class = "pill-good"
    else:
        citations_value = "0 cited"
        citations_class = "pill-neutral"

    st.markdown(f"""
    <div class="trust-strip">
        <div class="trust-pill {'pill-good' if grounded else 'pill-neutral'}">
            <div class="label">Grounded</div><div class="value">{'Yes' if grounded else 'No (baseline)' if not result['use_rag'] else 'No relevant sources'}</div>
        </div>
        <div class="trust-pill {citations_class}">
            <div class="label">Citations</div><div class="value">{citations_value}</div>
        </div>
        <div class="trust-pill {'pill-warn' if hedged else 'pill-neutral'}">
            <div class="label">Hedged</div><div class="value">{'Yes' if hedged else 'No'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def run_query(model_name, rag_flag):
    return trustsoc_answer(q, model=model_name, use_rag=rag_flag, k=k, threshold=threshold, balance=balance)

def render_full(result):
    render_answer_panel(result)
    render_sources_panel(result)
    render_citation_ledger(result)
    render_trust_strip(result)

# ---------------------------------------------------------------------------
# Main render — builds every requested (model, RAG) combination and lays
# them out as side-by-side columns, so either axis (model, RAG on/off) or
# both can be compared at once without duplicating the render logic.
# ---------------------------------------------------------------------------
if run and q:
    models_to_run = [MODEL_A, MODEL_B] if compare_models else [model]
    rag_settings = [True, False] if compare_rag else [use_rag]
    combos = [(m, r) for m in models_to_run for r in rag_settings]

    if len(combos) == 1:
        result = run_query(*combos[0])
        c1, c2 = st.columns([3, 2])
        with c1:
            render_answer_panel(result)
            render_trust_strip(result)
        with c2:
            render_sources_panel(result)
            render_citation_ledger(result)
    else:
        cols = st.columns(len(combos))
        for col, (m, r) in zip(cols, combos):
            with col:
                render_full(run_query(m, r))
