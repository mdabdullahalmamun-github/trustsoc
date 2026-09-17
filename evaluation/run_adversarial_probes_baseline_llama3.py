"""
The RAG-vs-non-RAG baseline for the 30 adversarial probes.

Important methodological note, worth reading before trusting the numbers
this produces: the RAG condition's hedge detection (run_adversarial_probes_
mistral.py / _llama3.py) checks for an EXACT phrase match against
INSUFFICIENT_CONTEXT_MSG, because the RAG system prompt explicitly
instructs the model to use that exact sentence (Rule 3). The no-RAG
baseline prompt has no such instruction at all -- it is just "You are a
SOC analyst assistant. Answer concisely." -- so exact-phrase matching
would be a meaningless, near-certain-to-fail comparison here, not a fair
test of whether the model hedges.

This script therefore checks TWO things separately:
  exact_rag_style_hedge  -- does it happen to match the RAG condition's
                             exact phrase anyway (checked for direct
                             comparability, but not expected to fire often)
  general_hedge_detected -- does the answer contain ANY of a reasonable
                             set of common uncertainty/refusal phrases
                             (a heuristic, not a formal instruction match)
Both are reported, alongside the raw answer text, so you can read the
actual difference yourself rather than trust either number blindly --
consistent with how every other automated classification in this project
has needed manual verification before being reported as a finding.

Place this in your evaluation/ folder alongside adversarial_probes.json.

Run from your project root:
    python evaluation/run_adversarial_probes_baseline_mistral.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_pipeline import trustsoc_answer, INSUFFICIENT_CONTEXT_MSG

MODEL = "llama3"
RESULTS_FILENAME = "adversarial_probes_results_baseline_llama3.json"

import re

# A deliberately broad, heuristic set -- not a formal instruction the model
# was given, just common ways a model might express uncertainty unprompted.
# Flexible regex patterns are used (rather than exact substrings) so that
# natural variation -- "I don't have any SPECIFIC information", "I don't
# have information" -- is still caught without needing to enumerate every
# possible insertion.
_HEDGE_PATTERNS = [
    r"\bi\s+(don't|do not)\s+know\b",
    r"\b(not|isn't|is not)\s+aware\b",
    r"\bi\s+(don't|do not)\s+have\s+(any\s+)?(specific\s+)?information\b",
    r"\bno\s+information\b",
    r"\bi\s+(cannot|can't)\s+find\b",
    r"\bunable\s+to\b",
    r"\bnot\s+familiar\s+with\b",
    r"\bno\s+record\b",
    r"\b(cannot|can't)\s+(determine|confirm|verify)\b",
    r"\bnot\s+certain\b",
    r"\b(does|doesn't|does not)\s+exist\b",
    r"\bno\s+such\b",
    r"\bnot\s+a\s+(real|known)\b",
]

def general_hedge_detected(answer_text):
    lower = answer_text.lower()
    return any(re.search(pattern, lower) for pattern in _HEDGE_PATTERNS)


def main():
    probes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adversarial_probes.json")
    with open(probes_path, encoding="utf-8") as f:
        probes = json.load(f)

    results = []
    for p in probes:
        answer = trustsoc_answer(p["question"], model=MODEL, use_rag=False)
        answer_text = answer.get("answer", "")

        exact_hedge = INSUFFICIENT_CONTEXT_MSG in answer_text
        general_hedge = general_hedge_detected(answer_text)

        results.append({
            "id": p["id"],
            "category": p["category"],
            "question": p["question"],
            "answer": answer_text,
            "exact_rag_style_hedge": exact_hedge,
            "general_hedge_detected": general_hedge,
        })
        print(f"{p['id']}  [{p['category']}]  exact_hedge={exact_hedge}  general_hedge={general_hedge}")

    total = len(results)
    exact_count = sum(1 for r in results if r["exact_rag_style_hedge"])
    general_count = sum(1 for r in results if r["general_hedge_detected"])
    neither_count = sum(1 for r in results if not r["exact_rag_style_hedge"] and not r["general_hedge_detected"])

    print(f"\n=== {MODEL} NO-RAG BASELINE on adversarial probes (n={total}) ===")
    print(f"Exact RAG-style hedge phrase present: {exact_count}/{total}")
    print(f"General hedge language detected (heuristic): {general_count}/{total}")
    print(f"Neither -- likely a confident, unhedged answer: {neither_count}/{total}")
    print("\nRead a sample of the 'neither' cases directly before treating this as a finding --")
    print("this heuristic is not a validated classifier, only a starting signal.")

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILENAME)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-item results (including full answer text) written to {results_path}")


if __name__ == "__main__":
    main()
