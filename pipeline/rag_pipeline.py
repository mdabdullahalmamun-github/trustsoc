"""Core RAG pipeline: the citation-enforcing prompt, the Phase 2.5 source-balanced
retriever, and trustsoc_answer() — the single function used by every entry point
(demo/Live_Demo.py, ui/dashboard.py, demo/test_retrieval.py) to query either model
with or without retrieval."""

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

# ---- the Phase 2.5 balanced retriever ----
collection = chromadb.PersistentClient(path="kb").get_collection("cti")
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_bge_st = SentenceTransformer("BAAI/bge-small-en-v1.5")

def retrieve(query, k=6, balance=True):
    q_emb = _bge_st.encode(_BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    def _run(where=None, n=k):
        r = collection.query(query_embeddings=[q_emb], n_results=n, where=where)
        return [{"text": doc, "source": md.get("source"),
                 "id": md.get("technique_id") or md.get("cve_id"),
                 "distance": round(dist, 3)}
                for doc, md, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

    if not balance:
        return _run(n=k)

    half = max(1, k // 2)
    attack = _run(where={"source": "MITRE ATT&CK"}, n=half)
    others = _run(where={"source": {"$ne": "MITRE ATT&CK"}}, n=k - half)
    return sorted(attack + others, key=lambda h: h["distance"])[:k]

# ---- 3.2: the pipeline — one function, both models, RAG on/off ----
MODEL_A = "mistral"
MODEL_B = "llama3"

def generate(model, prompt, temperature=0.0, retries=1):
    """One deterministic generation from a local Ollama model, with one automatic retry
    if Ollama fails to return a proper response (occasional GPU/VRAM hiccup on 4GB cards)."""
    for attempt in range(retries + 1):
        r = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}}, timeout=180)
        data = r.json()
        if "response" in data:
            return data["response"].strip()
        if attempt < retries:
            time.sleep(2)
            continue
        return f"[Ollama did not return a response after retry: {data.get('error', 'unknown')}]"
    return "[unexpected error]"

def trustsoc_answer(question, model=MODEL_A, use_rag=True, k=5):
    """Return a dict with the answer, the contexts used, and timing."""
    t0 = time.time()
    if use_rag:
        hits = retrieve(question, k=k)
        contexts = [h["text"] for h in hits]
        prompt = build_prompt(question, contexts)
    else:
        contexts = []
        prompt = f"You are a SOC analyst assistant. Answer concisely.\n\nQUESTION: {question}\n\nANSWER:"
    answer = generate(model, prompt)
    return {"question": question, "model": model, "use_rag": use_rag,
            "answer": answer, "contexts": contexts, "seconds": round(time.time()-t0, 2)}

if __name__ == "__main__":
    demo = trustsoc_answer("Which ATT&CK technique covers malicious PowerShell use?", MODEL_A, True)
    print(demo["answer"])
    print("---", demo["seconds"], "s, contexts used:", len(demo["contexts"]))