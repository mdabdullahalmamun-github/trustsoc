"""
Run soc_scenarios.json through the TrustSOC pipeline and score the results.
Place this in your evaluation/ folder alongside soc_scenarios.json.

Run it from your project root:
    python evaluation/run_soc_scenarios.py
"""
import json
import sys
import os

# Make sure the project root (the folder containing pipeline/) is on the
# import path, regardless of which folder this script is actually run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer


def score_answer(cited_technique_ids, ground_truth_id):
    """
    Matches the scoring rubric already described in Chapter 3.5:
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
    # Load the scenarios file relative to this script's own folder, so it
    # works no matter what directory you run the command from.
    scenarios_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soc_scenarios.json")
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    for s in scenarios:
        answer = trustsoc_answer(s["question"], model="mistral", k=5)
        cited = answer["citation_check"]["cited"]

        score = score_answer(cited, s["ground_truth_technique"])
        citation_check = answer.get("citation_check", {})
        results.append({
            "id": s["id"],
            "category": s["category"],
            "ground_truth": s["ground_truth_technique"],
            "cited": cited,
            "score": score,
            # New fields, now that rag_pipeline.py's validate_citations() also
            # flags bracketed references that don't use the required ID format
            # (e.g. "[Citation: ProofPoint Serpent]") -- captured here across
            # all 50 questions, not just the ones checked manually.
            "fabricated": citation_check.get("fabricated", []),
            "unrecognised_citations": citation_check.get("unrecognised_citations", []),
        })
        print(f"{s['id']}  score={score}  gt={s['ground_truth_technique']}  cited={cited}")

    # Summary
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["score"])

    print(f"\nOverall average score: {avg_score:.3f} across {total} scenarios\n")
    print("By category:")
    for cat, scores in by_category.items():
        print(f"  {cat}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")

    fabricated_count = sum(1 for r in results if r["fabricated"])
    unrecognised_count = sum(1 for r in results if r["unrecognised_citations"])
    print(f"\nScenarios with at least one fabricated citation: {fabricated_count}/{total}")
    print(f"Scenarios with at least one unrecognised-format citation: {unrecognised_count}/{total}")

    # Save full per-item results for later analysis / Chapter 5 tables
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soc_scenarios_results_mistral.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
