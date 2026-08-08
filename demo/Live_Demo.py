"""Interactive terminal demo. For each question typed, shows the answer with
and without RAG side by side — the clearest way to demonstrate that grounding
in the knowledge base fixes ungrounded hallucination (see build log, Phase 0.11)."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import trustsoc_answer

print("TrustSOC live demo — type a SOC question (or 'quit')\n")
while True:
    q = input("Question: ").strip()
    if q.lower() in ("quit", "exit", ""):
        break

    print("\n--- WITHOUT RAG ---")
    r1 = trustsoc_answer(q, model="mistral", use_rag=False)
    print(r1["answer"])

    print("\n--- WITH RAG ---")
    r2 = trustsoc_answer(q, model="mistral", use_rag=True)
    print(r2["answer"])
    print(f"\n(grounded in {len(r2['contexts'])} retrieved sources, {r2['seconds']}s)\n")
    print("-"*70 + "\n")