"""
The RAG-vs-non-RAG baseline for the 50 custom SOC scenarios -- completes
the Faithfulness dimension's baseline comparison alongside the CTIBench-ATE
baseline, together covering the full query set your proposal's Phase 5
task specifies ("all queries x 2 models x 2 conditions").

Uses the same single-answer scoring rubric as the RAG condition
(run_soc_scenarios_mistral.py / _llama3.py): 1.0 exact sub-technique
match, 0.5 correct parent technique, 0.0 otherwise -- so the two
conditions are directly comparable. As with the CTIBench-ATE baseline,
citation_check is None without retrieval, so technique IDs are extracted
directly from the raw answer text.

Place this in your evaluation/ folder alongside soc_scenarios.json.

Run from your project root:
    python evaluation/run_soc_scenarios_baseline_mistral.py
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer

MODEL = "mistral"
RESULTS_FILENAME = "soc_scenarios_results_baseline_mistral.json"

_TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def score_answer(cited_technique_ids, ground_truth_id):
    """Same rubric as the RAG condition (Chapter 3.5):
      1.0 = exact sub-technique match
      0.5 = correct parent technique, wrong (or missing) sub-technique
      0.0 = no match at all
    """
    if ground_truth_id in cited_technique_ids:
        return 1.0
    gt_parent = ground_truth_id.split(".")[0]
    cited_parents = {t.split(".")[0] for t in cited_technique_ids}
    if gt_parent in cited_parents:
        return 0.5
    return 0.0


def main():
    scenarios_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soc_scenarios.json")
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    for s in scenarios:
        answer = trustsoc_answer(s["question"], model=MODEL, use_rag=False)
        answer_text = answer.get("answer", "")
        cited = sorted(set(_TECHNIQUE_ID_RE.findall(answer_text)))

        score = score_answer(cited, s["ground_truth_technique"])
        results.append({
            "id": s["id"],
            "category": s["category"],
            "ground_truth": s["ground_truth_technique"],
            "cited": cited,
            "score": score,
        })
        print(f"{s['id']}  score={score}  gt={s['ground_truth_technique']}  cited={cited}")

    total = len(results)
    avg_score = sum(r["score"] for r in results) / total
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["score"])

    print(f"\n=== {MODEL} NO-RAG BASELINE on SOC scenarios (n={total}) ===")
    print(f"Overall average score: {avg_score:.3f}\n")
    print("By category:")
    for cat, scores in by_category.items():
        print(f"  {cat}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")
    print("\nCompare directly against soc_scenarios_results_mistral.json (the RAG condition)")
    print("for the RQ1 statistical test.")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
