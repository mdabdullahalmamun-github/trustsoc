"""
The RAG-vs-non-RAG baseline for CTIBench-ATE -- directly answering RQ1 as
your proposal literally words it: "Can a RAG-augmented LLM... produce
significantly more faithful and traceable threat analysis outputs than
a non-RAG baseline, measured on the CTIBench-ATE benchmark?"

Uses trustsoc_answer(..., use_rag=False), which was already implemented
in the pipeline but never run as a full evaluation condition. With RAG
disabled, there is no retrieved context to check citations against, so
citation_check is None -- technique IDs are instead extracted directly
from the raw answer text using the same regex pattern rag_pipeline.py
itself uses, and scored with the same multi-label precision/recall/F1
as the RAG condition (run_ctibench_ate_mistral.py / _llama3.py), so the
two conditions are directly comparable.

Place this in your evaluation/ folder alongside ctibench_ate.json.

Run from your project root:
    python evaluation/run_ctibench_ate_baseline_mistral.py
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer

MODEL = "mistral"
RESULTS_FILENAME = "ctibench_ate_results_baseline_mistral.json"

# Same pattern rag_pipeline.py's validate_citations() uses -- reused here
# since there is no retrieved context to validate against in the baseline
# condition, only the model's own unaided answer to extract IDs from.
_TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def to_parent(technique_id):
    return technique_id.split(".")[0]


def score_item(cited_ids, ground_truth_ids):
    predicted_parents = {to_parent(t) for t in cited_ids}
    gt_parents = set(ground_truth_ids)
    tp = len(predicted_parents & gt_parents)
    fp = len(predicted_parents - gt_parents)
    fn = len(gt_parents - predicted_parents)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctibench_ate.json")
    with open(data_path, encoding="utf-8") as f:
        items = json.load(f)

    results = []
    for it in items:
        answer = trustsoc_answer(it["question"], model=MODEL, use_rag=False)
        answer_text = answer.get("answer", "")
        cited = sorted(set(_TECHNIQUE_ID_RE.findall(answer_text)))

        scores = score_item(cited, it["ground_truth_techniques"])

        results.append({
            "id": it["id"],
            "mitre_software_id": it["mitre_software_id"],
            "ground_truth_techniques": it["ground_truth_techniques"],
            "cited": cited,
            **scores,
        })
        print(f"{it['id']} ({it['mitre_software_id']})  "
              f"P={scores['precision']:.2f} R={scores['recall']:.2f} F1={scores['f1']:.2f}")

    total = len(results)
    avg_p = sum(r["precision"] for r in results) / total
    avg_r = sum(r["recall"] for r in results) / total
    avg_f1 = sum(r["f1"] for r in results) / total

    print(f"\n=== {MODEL} NO-RAG BASELINE on CTIBench-ATE (Enterprise-only, n={total}) ===")
    print(f"Mean precision: {avg_p:.3f}")
    print(f"Mean recall:    {avg_r:.3f}")
    print(f"Mean F1:        {avg_f1:.3f}")
    print("\nCompare directly against ctibench_ate_results_mistral.json (the RAG condition)")
    print("for the RQ1 statistical test.")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results written to {results_path}")


if __name__ == "__main__":
    main()
