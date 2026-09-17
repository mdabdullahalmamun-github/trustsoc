"""
The RAG-vs-non-RAG baseline for the 50 paraphrase pairs -- completes the
Consistency dimension's baseline comparison alongside the SOC scenario
and CTIBench-ATE baselines, matching the proposal's evaluation design
table, which specifies "RAG vs non-RAG" as a baseline comparison for
Consistency as well as Faithfulness.

Uses the exact same cosine-similarity + technique-agreement scoring as
the RAG condition (run_paraphrase_pairs_mistral.py / _llama3.py), so the
two conditions are directly comparable -- no special-casing needed here
the way the adversarial probes required, since Consistency compares two
answers to each other rather than checking for a specific hedge phrase.

Place this in your evaluation/ folder alongside paraphrase_pairs.json.

Run from your project root:
    python evaluation/run_paraphrase_pairs_baseline_mistral.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL = "mistral"
RESULTS_FILENAME = "paraphrase_pairs_results_baseline_mistral.json"

_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paraphrase_pairs.json")
    with open(data_path, encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    for pair in pairs:
        answer_a = trustsoc_answer(pair["phrasing_a"], model=MODEL, use_rag=False)
        answer_b = trustsoc_answer(pair["phrasing_b"], model=MODEL, use_rag=False)

        text_a = answer_a.get("answer", "")
        text_b = answer_b.get("answer", "")

        emb_a = _embedder.encode(text_a, normalize_embeddings=True)
        emb_b = _embedder.encode(text_b, normalize_embeddings=True)
        similarity = float(cos_sim(emb_a, emb_b)[0][0])

        # No citation_check without retrieval -- extract technique IDs
        # directly from the raw answer text, same pattern as the other
        # baseline scripts.
        import re
        _TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
        cited_a = set(_TECHNIQUE_ID_RE.findall(text_a))
        cited_b = set(_TECHNIQUE_ID_RE.findall(text_b))

        gt = pair["ground_truth_technique"]
        gt_parent = gt.split(".")[0]
        a_parents = {t.split(".")[0] for t in cited_a}
        b_parents = {t.split(".")[0] for t in cited_b}
        technique_agreement = bool(a_parents & b_parents)
        both_correct = (gt_parent in a_parents) and (gt_parent in b_parents)

        results.append({
            "id": pair["id"],
            "category": pair["category"],
            "ground_truth_technique": gt,
            "answer_a": text_a,
            "answer_b": text_b,
            "cited_a": sorted(cited_a),
            "cited_b": sorted(cited_b),
            "cosine_similarity": round(similarity, 4),
            "technique_agreement": technique_agreement,
            "both_correct": both_correct,
        })
        print(f"{pair['id']}  similarity={similarity:.3f}  "
              f"technique_agreement={technique_agreement}  both_correct={both_correct}")

    total = len(results)
    avg_sim = sum(r["cosine_similarity"] for r in results) / total
    agree_count = sum(1 for r in results if r["technique_agreement"])

    print(f"\n=== {MODEL} NO-RAG BASELINE Consistency (n={total} pairs) ===")
    print(f"Mean cosine similarity: {avg_sim:.3f}")
    print(f"Technique agreement: {agree_count}/{total} ({100*agree_count/total:.0f}%)")
    print("\nCompare directly against paraphrase_pairs_results_mistral.json (the RAG condition).")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
