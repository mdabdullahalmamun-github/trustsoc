"""
Run ctibench_ate.json through the TrustSOC pipeline and score with
multi-label precision/recall/F1 -- NOT the single-answer 1.0/0.5/0.0
rubric used for the SOC scenarios, since ground truth here is a SET of
techniques per item (avg 6.4, up to 11), not one.

Scoring detail: cited technique IDs are normalised to their PARENT
technique (stripping any .XXX sub-technique suffix) before comparison,
since CTIBench's own published ground truth is parent-level only. This
matches how the original CTIBench paper evaluates this task, and avoids
penalising TrustSOC for correctly citing a more specific sub-technique
than CTIBench's ground truth happens to record.

Place this in your evaluation/ folder alongside ctibench_ate.json.

Run it from your project root:
    python evaluation/run_ctibench_ate_mistral.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer

MODEL = "llama3"
RESULTS_FILENAME = "ctibench_ate_results_llama3.json"


def to_parent(technique_id):
    return technique_id.split(".")[0]


def score_item(cited_ids, ground_truth_ids):
    """Multi-label precision / recall / F1 at parent-technique level."""
    predicted_parents = {to_parent(t) for t in cited_ids if t.startswith("T")}
    gt_parents = set(ground_truth_ids)  # already parent-level in CTIBench's GT

    tp = len(predicted_parents & gt_parents)
    fp = len(predicted_parents - gt_parents)
    fn = len(gt_parents - predicted_parents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": sorted(predicted_parents & gt_parents),
        "false_positives": sorted(predicted_parents - gt_parents),
        "false_negatives": sorted(gt_parents - predicted_parents),
    }


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctibench_ate.json")
    with open(data_path, encoding="utf-8") as f:
        items = json.load(f)

    results = []
    for it in items:
        answer = trustsoc_answer(it["question"], model=MODEL, k=5)
        citation_check = answer.get("citation_check") or {}
        cited = citation_check.get("cited", [])

        scores = score_item(cited, it["ground_truth_techniques"])

        results.append({
            "id": it["id"],
            "mitre_software_id": it["mitre_software_id"],
            "ground_truth_techniques": it["ground_truth_techniques"],
            "cited": cited,
            "fabricated": citation_check.get("fabricated", []),
            "unrecognised_citations": citation_check.get("unrecognised_citations", []),
            **scores,
        })
        print(f"{it['id']} ({it['mitre_software_id']})  "
              f"P={scores['precision']:.2f} R={scores['recall']:.2f} F1={scores['f1']:.2f}  "
              f"gt_size={len(it['ground_truth_techniques'])}")

    total = len(results)
    avg_p = sum(r["precision"] for r in results) / total
    avg_r = sum(r["recall"] for r in results) / total
    avg_f1 = sum(r["f1"] for r in results) / total

    print(f"\n=== {MODEL} on CTIBench-ATE (Enterprise-only, n={total}) ===")
    print(f"Mean precision: {avg_p:.3f}")
    print(f"Mean recall:    {avg_r:.3f}")
    print(f"Mean F1:        {avg_f1:.3f}")

    fab_count = sum(1 for r in results if r["fabricated"])
    unrec_count = sum(1 for r in results if r["unrecognised_citations"])
    print(f"\nItems with >=1 fabricated citation: {fab_count}/{total}")
    print(f"Items with >=1 unrecognised-format citation: {unrec_count}/{total}")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
