"""Core RAG pipeline: the citation-enforcing prompt, the Phase 2.5 source-balanced
retriever, citation validation, and trustsoc_answer() — the single function used by
every entry point (demo/Live_Demo.py, ui/dashboard.py, check_retrieval.py) to
query either model with or without retrieval."""

import re
import requests, time
import chromadb
from sentence_transformers import SentenceTransformer

# ---- 3.1: the citation-enforcing prompt ----
SYSTEM_PROMPT = """You are a Security Operations Centre (SOC) analyst assistant.
Answer the analyst's question using ONLY the CONTEXT provided below.

Rules:
1. Use only facts found in the CONTEXT. Do not use outside knowledge.
2. After each claim, cite the source in square brackets, e.g. [MITRE ATT&CK T1059].
3. If the CONTEXT does not contain enough information to answer, reply exactly:
   "The provided context does not contain enough information to answer this reliably."
4. Do not invent technique IDs, CVE numbers, or threat actor names.
Be concise and precise."""

def build_prompt(question, contexts):
    ctx = "\n\n".join(f"- {c}" for c in contexts)
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"

# The exact refusal string the prompt instructs the model to use when context
# is insufficient. Reused below so retrieve() and trustsoc_answer() agree on
# what "no relevant context" means, instead of the model being the only judge.
INSUFFICIENT_CONTEXT_MSG = "The provided context does not contain enough information to answer this reliably."

# ---- the Phase 2.5 balanced retriever ----
collection = chromadb.PersistentClient(path="kb").get_collection("cti")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_bge_st = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ---- Fix 1: relevance threshold ----
# Cosine distance cutoff (this collection uses hnsw:space="cosine" — see build_kb.py).
# 0.0 = identical, larger = less similar. Chunks beyond this are treated as not
# actually relevant, rather than being handed to the model just because they were
# the "closest" of a bad set. Previously retrieve() always returned exactly k
# chunks regardless of quality, so the model's refusal (Rule 3 above) was the
# only safeguard against low-quality context — this threshold makes retrieval
# itself responsible for recognising when nothing useful was found.
#
# NOTE: 0.5 is a reasonable starting point for BGE-small cosine distance, not an
# empirically validated value. Calibrate this during Phase 4 using the planned
# ~20 hand-labelled answers (see build notebook, Phase 4 plan) — check whether
# chunks above/below this cutoff are actually relevant, and adjust accordingly.
# Document whatever value you settle on, and why, in Chapter 4.
RELEVANCE_THRESHOLD = 0.5

def retrieve(query, k=5, balance=True, threshold=RELEVANCE_THRESHOLD, fetch_multiplier=4):
    """Retrieve up to k chunks, source-balanced by default, filtered to those
    within `threshold` cosine distance. May return fewer than k chunks — or
    zero — if nothing sufficiently relevant was found. Pass threshold=None to
    disable filtering (e.g. for the retrieval-quality sanity check).

    When filtering is active, a wider pool (n * fetch_multiplier) is fetched
    from ChromaDB before the threshold cut and trim to k — otherwise a
    relevant chunk ranked just outside the first `n` results would never be
    considered, even if it would have passed the threshold."""
    q_emb = _bge_st.encode(_BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    def _run(where=None, n=k):
        # Guard against n<=0 (e.g. k=1 makes the "others" half of the split
        # request 0 results) — ChromaDB errors on n_results<=0 rather than
        # returning an empty list, so this must be caught before the query,
        # not after. Reachable from the dashboard slider (min value is 1).
        if n <= 0:
            return []
        fetch_n = n * fetch_multiplier if threshold is not None else n
        r = collection.query(query_embeddings=[q_emb], n_results=fetch_n, where=where)
        hits = [{"text": doc, "source": md.get("source"),
                 "id": md.get("technique_id") or md.get("cve_id"),
                 "distance": round(dist, 3)}
                for doc, md, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        if threshold is not None:
            hits = [h for h in hits if h["distance"] <= threshold]
        return hits[:n]

    if not balance:
        return _run(n=k)

    half = max(1, k // 2)
    attack = _run(where={"source": "MITRE ATT&CK"}, n=half)
    others = _run(where={"source": {"$ne": "MITRE ATT&CK"}}, n=k - half)
    return sorted(attack + others, key=lambda h: h["distance"])[:k]

# ---- Fix 2: citation validation ----
# Extracts technique IDs (Txxxx or Txxxx.xxx) and CVE IDs from the model's answer,
# and checks each against the IDs actually present in the retrieved context.
# This is what turns Rule 4 of the prompt ("do not invent IDs") from an
# instruction the model might ignore into something you can actually measure —
# feed this into your Phase 4 faithfulness scorer alongside the CTIBench-ATE
# ground-truth matching.
_TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_CVE_ID_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b")

def validate_citations(answer, contexts_meta):
    """Return which IDs cited in `answer` were actually present in the retrieved
    context (grounded) vs not (fabricated). `contexts_meta` is the list of hit
    dicts returned by retrieve() (needs the 'id' field)."""
    retrieved_ids = {h["id"] for h in contexts_meta if h.get("id")}
    cited_ids = set(_TECHNIQUE_ID_RE.findall(answer)) | set(_CVE_ID_RE.findall(answer))
    grounded = cited_ids & retrieved_ids
    fabricated = cited_ids - retrieved_ids
    return {
        "cited": sorted(cited_ids),
        "grounded": sorted(grounded),
        "fabricated": sorted(fabricated),
        "has_fabrication": bool(fabricated),
    }

# ---- 3.2: the pipeline — one function, both models, RAG on/off ----
MODEL_A = "mistral"
MODEL_B = "llama3"

def generate(model, prompt, temperature=0.0, retries=2):
    """One deterministic generation from a local Ollama model, with automatic retries
    if Ollama fails to return a proper response OR the connection itself fails
    (occasional GPU/VRAM hiccup on 4GB cards — see README known hardware limit).
    Never raises: always returns a string, so a single dropped connection during a
    Phase 4 batch run (hundreds of sequential calls) can't crash the whole run.
    Note: temperature=0 is near-deterministic, not guaranteed bit-identical across
    runs, due to GPU floating-point non-associativity — document this as a limitation."""
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            r = requests.post("http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": temperature}}, timeout=180)
            data = r.json()
            if "response" in data:
                return data["response"].strip()
            last_error = data.get("error", "unknown")
        except requests.exceptions.RequestException as e:
            # Connection refused, dropped, or timed out — Ollama not reachable
            # or crashed mid-request. Caught here specifically so a batch
            # evaluation run degrades to a logged failure on this one query
            # instead of stopping entirely.
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2)
    return f"[GENERATION FAILED after {retries + 1} attempts: {last_error}]"

def trustsoc_answer(question, model=MODEL_A, use_rag=True, k=5,
                     threshold=RELEVANCE_THRESHOLD, balance=True):
    """Return a dict with the answer, the contexts used, citation validation,
    and timing. If use_rag=True but nothing passes the relevance threshold,
    the model is not called at all — trustsoc_answer returns the standard
    insufficient-context message directly, so an ungrounded guess is never
    possible even if the model would have ignored Rule 3.

    threshold and balance are exposed here (rather than hardcoded) so callers
    like the dashboard can adjust them without duplicating retrieval logic —
    defaults match the pipeline's standard behaviour exactly."""
    t0 = time.time()
    citation_check = None
    hits = []
    if use_rag:
        hits = retrieve(question, k=k, threshold=threshold, balance=balance)
        contexts = [h["text"] for h in hits]
        if not hits:
            return {"question": question, "model": model, "use_rag": use_rag,
                    "answer": INSUFFICIENT_CONTEXT_MSG, "contexts": [], "hits": [],
                    "citation_check": None, "seconds": round(time.time() - t0, 2)}
        prompt = build_prompt(question, contexts)
    else:
        hits = []
        contexts = []
        prompt = f"You are a SOC analyst assistant. Answer concisely.\n\nQUESTION: {question}\n\nANSWER:"

    answer = generate(model, prompt)
    if use_rag:
        citation_check = validate_citations(answer, hits)

    return {"question": question, "model": model, "use_rag": use_rag,
            "answer": answer, "contexts": contexts, "hits": hits,
            "citation_check": citation_check, "seconds": round(time.time()-t0, 2)}

if __name__ == "__main__":
    demo = trustsoc_answer("Which ATT&CK technique covers malicious PowerShell use?", MODEL_A, True)
    print(demo["answer"])
    print("---", demo["seconds"], "s, contexts used:", len(demo["contexts"]))
    print("--- citation check:", demo["citation_check"])
