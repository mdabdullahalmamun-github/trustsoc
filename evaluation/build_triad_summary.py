"""
Generates the complete Chapter 5 Triad summary -- all three dimensions,
both models, from the real result files already produced by the Phase 4
evaluation scripts. Run this any time any of the underlying result files
change, rather than recomputing tables by hand.

Expects these files to exist in the results/ folder at the project root:
    soc_scenarios_results_mistral.json / _llama3.json
    ctibench_ate_results_mistral.json / _llama3.json
    adversarial_probes_results_mistral.json / _llama3.json
    paraphrase_pairs_results_mistral.json / _llama3.json

Run from your project root:
    python evaluation/build_triad_summary.py
"""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(PROJECT_ROOT, "results")


def load(filename):
    path = os.path.join(FOLDER, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def faithfulness_summary(model_name, soc_file, ate_file):
    soc = load(soc_file)
    ate = load(ate_file)
    lines = [f"--- Faithfulness: {model_name} ---"]
    if soc:
        avg = sum(d["score"] for d in soc) / len(soc)
        fab = sum(1 for d in soc if d.get("fabricated"))
        unrec = sum(1 for d in soc if d.get("unrecognised_citations"))
        lines.append(f"  SOC scenarios (n={len(soc)}): mean score={avg:.3f}  "
                      f"fabricated={fab}/{len(soc)}  unrecognised={unrec}/{len(soc)}")
    else:
        lines.append(f"  SOC scenarios: FILE NOT FOUND ({soc_file}) -- check it's in results/")
    if ate:
        avg_p = sum(d["precision"] for d in ate) / len(ate)
        avg_r = sum(d["recall"] for d in ate) / len(ate)
        avg_f1 = sum(d["f1"] for d in ate) / len(ate)
        fab_total = sum(len(d.get("fabricated", [])) for d in ate)
        lines.append(f"  CTIBench-ATE (n={len(ate)}): P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}  "
                      f"total fabricated citations={fab_total}")
    else:
        lines.append(f"  CTIBench-ATE: FILE NOT FOUND ({ate_file}) -- check it's in results/")
    return lines


def uncertainty_summary(model_name, adv_file):
    adv = load(adv_file)
    if not adv:
        return [f"--- Uncertainty Expression: {model_name} --- FILE NOT FOUND ({adv_file})"]
    total = len(adv)
    clean = sum(1 for d in adv if d["outcome"] == "clean_hedge")
    hedge_answer = sum(1 for d in adv if d["outcome"] == "hedge_then_answer")
    no_hedge = sum(1 for d in adv if d["outcome"] == "no_hedge_confident")
    return [f"--- Uncertainty Expression: {model_name} --- (n={total} adversarial probes)",
            f"  clean_hedge={clean}/{total} ({100*clean/total:.0f}%)  "
            f"hedge_then_answer={hedge_answer}/{total} ({100*hedge_answer/total:.0f}%)  "
            f"no_hedge_confident={no_hedge}/{total} ({100*no_hedge/total:.0f}%)"]


def consistency_summary(model_name, para_file):
    para = load(para_file)
    if not para:
        return [f"--- Consistency: {model_name} --- FILE NOT FOUND ({para_file})"]
    total = len(para)
    avg_sim = sum(d["cosine_similarity"] for d in para) / total
    agree = sum(1 for d in para if d["technique_agreement"])
    return [f"--- Consistency: {model_name} --- (n={total} paraphrase pairs)",
            f"  mean cosine similarity={avg_sim:.3f}  technique_agreement={agree}/{total} ({100*agree/total:.0f}%)"]


def main():
    print("=" * 70)
    print("TRUSTSOC TRIAD -- FULL PHASE 4 SUMMARY")
    print("=" * 70)
    print()

    for model, soc_f, ate_f, adv_f, para_f in [
        ("Mistral", "soc_scenarios_results_mistral.json", "ctibench_ate_results_mistral.json",
         "adversarial_probes_results_mistral.json", "paraphrase_pairs_results_mistral.json"),
        ("LLaMA 3", "soc_scenarios_results_llama3.json", "ctibench_ate_results_llama3.json",
         "adversarial_probes_results_llama3.json", "paraphrase_pairs_results_llama3.json"),
    ]:
        print(f"\n{'='*20} {model} {'='*20}")
        for line in faithfulness_summary(model, soc_f, ate_f):
            print(line)
        print()
        for line in uncertainty_summary(model, adv_f):
            print(line)
        print()
        for line in consistency_summary(model, para_f):
            print(line)


if __name__ == "__main__":
    main()
