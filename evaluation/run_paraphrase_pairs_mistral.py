"""
Run paraphrase_pairs.json through the TrustSOC pipeline -- the Consistency
dimension of the Triad (Chapter 3.5): for each pair, both phrasings are run
through the SAME model, and the two answers are compared two ways:

1. Cosine similarity between the two answers' embeddings, using the same
   BGE-small-en model already used throughout the project (no new
   dependency, and directly comparable to the retrieval embeddings elsewhere).
2. technique_agreement -- did both phrasings cite the same ground-truth
   technique, regardless of what else they said? A stricter, more
   interpretable companion to the similarity score: two answers can be
   textually quite different (different wording, different elaboration)
   while still agreeing on the one thing that actually matters -- which
   technique this is.

Place this in your evaluation/ folder alongside paraphrase_pairs.json.

Run it from your project root:
    python evaluation/run_paraphrase_pairs_mistral.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL = "mistral"
RESULTS_FILENAME = "paraphrase_pairs_results_mistral.json"

# Same embedding model already used for retrieval elsewhere in the project --
# deliberately not re-using the BGE query-instruction prefix here, since
# these are two generated answers being compared to each other, not a
# query being matched against passages.
_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paraphrase_pairs.json")
    with open(data_path, encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    for pair in pairs:
        answer_a = trustsoc_answer(pair["phrasing_a"], model=MODEL, k=5)
        answer_b = trustsoc_answer(pair["phrasing_b"], model=MODEL, k=5)

        text_a = answer_a.get("answer", "")
        text_b = answer_b.get("answer", "")

        emb_a = _embedder.encode(text_a, normalize_embeddings=True)
        emb_b = _embedder.encode(text_b, normalize_embeddings=True)
        similarity = float(cos_sim(emb_a, emb_b)[0][0])

        cited_a = set((answer_a.get("citation_check") or {}).get("cited", []))
        cited_b = set((answer_b.get("citation_check") or {}).get("cited", []))
        gt = pair["ground_truth_technique"]
        # Parent-level agreement check -- do both phrasings' citations agree
        # on the same technique, regardless of sub-technique specificity.
        gt_parent = gt.split(".")[0]
        a_parents = {t.split(".")[0] for t in cited_a}
        b_parents = {t.split(".")[0] for t in cited_b}
        technique_agreement = bool(a_parents & b_parents)  # did they agree with EACH OTHER
        both_correct = (gt_parent in a_parents) and (gt_parent in b_parents)

        results.append({
            "id": pair["id"],
            "category": pair["category"],
            "ground_truth_technique": gt,
            "phrasing_a": pair["phrasing_a"],
            "phrasing_b": pair["phrasing_b"],
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
    both_correct_count = sum(1 for r in results if r["both_correct"])

    print(f"\n=== {MODEL} Consistency summary (n={total} pairs) ===")
    print(f"Mean cosine similarity: {avg_sim:.3f}")
    print(f"Technique agreement (both phrasings cite the same technique as each other): {agree_count}/{total} ({100*agree_count/total:.0f}%)")
    print(f"Both phrasings correct (agree AND match ground truth): {both_correct_count}/{total} ({100*both_correct_count/total:.0f}%)")

    print("\nBy category:")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, items in by_cat.items():
        n = len(items)
        sim = sum(it["cosine_similarity"] for it in items) / n
        agr = sum(1 for it in items if it["technique_agreement"])
        print(f"  {cat}: mean_similarity={sim:.3f}  technique_agreement={agr}/{n}")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
