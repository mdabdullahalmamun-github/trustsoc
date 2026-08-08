"""Interactive terminal demo. For each question typed, shows four answers:
both models (Mistral 7B, LLaMA 3 8B), each with and without RAG — the
clearest way to demonstrate both that grounding fixes hallucination AND
how the two models differ in trustworthiness for the same question (RQ3).
Also surfaces the two trustworthiness safeguards added to the pipeline:
relevance-filtered retrieval (no answer is generated from irrelevant context)
and citation validation (flags any technique/CVE ID the model cites that
wasn't actually in the retrieved context).

Note: four sequential model calls per question means this is noticeably
slower than a single-model demo — expect roughly 30-40s per question on
the documented hardware (RTX 3050 Laptop, 4GB VRAM), more if llama3 is
under memory pressure. This script is for side-by-side comparison, not
for quick iteration — use check_retrieval.py for fast retrieval-only checks."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import trustsoc_answer, MODEL_A, MODEL_B

MODELS = [MODEL_A, MODEL_B]

def print_result(label, result):
    print(f"\n--- {label} ---")
    print(result["answer"])

    if result["use_rag"]:
        if not result["contexts"]:
            print(f"(no chunks passed the relevance threshold — refused rather than guessing, {result['seconds']}s)")
        else:
            print(f"({len(result['contexts'])} sources retrieved, {result['seconds']}s)")
            cc = result["citation_check"]
            if cc and cc["cited"]:
                status = "all grounded" if not cc["has_fabrication"] else "** FABRICATED CITATION **"
                print(f"citations: {cc['cited']} — {status}")
                if cc["has_fabrication"]:
                    print(f"  fabricated: {cc['fabricated']}")
            else:
                print("citations: none cited in the answer")
    else:
        print(f"({result['seconds']}s, no retrieval)")

print("TrustSOC live demo — both models, with and without RAG (or 'quit')\n")
while True:
    try:
        q = input("Question: ").strip()
    except (KeyboardInterrupt, EOFError):
        break
    if q.lower() in ("quit", "exit", ""):
        break

    for model in MODELS:
        print(f"\n{'='*70}\nMODEL: {model}\n{'='*70}")
        print_result(f"{model} — WITHOUT RAG", trustsoc_answer(q, model=model, use_rag=False))
        print_result(f"{model} — WITH RAG", trustsoc_answer(q, model=model, use_rag=True))

    print("\n" + "-"*70 + "\n")
