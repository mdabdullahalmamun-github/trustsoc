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

Produces two things:
    1. The same plain-text summary printed to the console as before.
    2. A self-contained, styled HTML report written to
       results/triad_summary.html -- open it directly in any browser.
       No external files or internet connection needed; all styling is
       embedded in the one file.
"""
import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(PROJECT_ROOT, "results")


def load(filename):
    path = os.path.join(FOLDER, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Computation -- each function returns a plain dict of numbers, or None if
# the underlying file is missing. Both the console output and the HTML
# report are built from these same dicts, so the two can never disagree.
# ---------------------------------------------------------------------------

def compute_soc(soc_file):
    soc = load(soc_file)
    if not soc:
        return None
    n = len(soc)
    return {
        "n": n,
        "mean_score": sum(d["score"] for d in soc) / n,
        "fabricated": sum(1 for d in soc if d.get("fabricated")),
        "unrecognised": sum(1 for d in soc if d.get("unrecognised_citations")),
    }


def compute_ate(ate_file):
    ate = load(ate_file)
    if not ate:
        return None
    n = len(ate)
    return {
        "n": n,
        "precision": sum(d["precision"] for d in ate) / n,
        "recall": sum(d["recall"] for d in ate) / n,
        "f1": sum(d["f1"] for d in ate) / n,
        "fabricated_total": sum(len(d.get("fabricated", [])) for d in ate),
    }


def compute_uncertainty(adv_file):
    adv = load(adv_file)
    if not adv:
        return None
    n = len(adv)
    return {
        "n": n,
        "clean_hedge": sum(1 for d in adv if d["outcome"] == "clean_hedge"),
        "hedge_then_answer": sum(1 for d in adv if d["outcome"] == "hedge_then_answer"),
        "no_hedge_confident": sum(1 for d in adv if d["outcome"] == "no_hedge_confident"),
    }


def compute_consistency(para_file):
    para = load(para_file)
    if not para:
        return None
    n = len(para)
    return {
        "n": n,
        "mean_cosine_similarity": sum(d["cosine_similarity"] for d in para) / n,
        "technique_agreement": sum(1 for d in para if d["technique_agreement"]),
    }


FILES = {
    "Mistral 7B": dict(
        soc="soc_scenarios_results_mistral.json",
        ate="ctibench_ate_results_mistral.json",
        adv="adversarial_probes_results_mistral.json",
        para="paraphrase_pairs_results_mistral.json",
    ),
    "LLaMA 3 8B": dict(
        soc="soc_scenarios_results_llama3.json",
        ate="ctibench_ate_results_llama3.json",
        adv="adversarial_probes_results_llama3.json",
        para="paraphrase_pairs_results_llama3.json",
    ),
}


def gather_all_results():
    """Load and compute every metric for both models up front, once."""
    results = {}
    for model, files in FILES.items():
        results[model] = {
            "soc": compute_soc(files["soc"]),
            "ate": compute_ate(files["ate"]),
            "adv": compute_uncertainty(files["adv"]),
            "para": compute_consistency(files["para"]),
        }
    return results


# ---------------------------------------------------------------------------
# Console rendering (unchanged in spirit from the original script)
# ---------------------------------------------------------------------------

def print_console_report(results):
    print("=" * 70)
    print("TRUSTSOC TRIAD -- FULL PHASE 4 SUMMARY")
    print("=" * 70)

    for model, r in results.items():
        print(f"\n{'=' * 20} {model} {'=' * 20}")

        print(f"--- Faithfulness: {model} ---")
        if r["soc"]:
            s = r["soc"]
            print(f"  SOC scenarios (n={s['n']}): mean score={s['mean_score']:.3f}  "
                  f"fabricated={s['fabricated']}/{s['n']}  unrecognised={s['unrecognised']}/{s['n']}")
        else:
            print("  SOC scenarios: FILE NOT FOUND -- check it's in results/")
        if r["ate"]:
            a = r["ate"]
            print(f"  CTIBench-ATE (n={a['n']}): P={a['precision']:.3f} R={a['recall']:.3f} "
                  f"F1={a['f1']:.3f}  total fabricated citations={a['fabricated_total']}")
        else:
            print("  CTIBench-ATE: FILE NOT FOUND -- check it's in results/")

        print()
        if r["adv"]:
            u = r["adv"]
            n = u["n"]
            print(f"--- Uncertainty Expression: {model} --- (n={n} adversarial probes)")
            print(f"  clean_hedge={u['clean_hedge']}/{n} ({100*u['clean_hedge']/n:.0f}%)  "
                  f"hedge_then_answer={u['hedge_then_answer']}/{n} ({100*u['hedge_then_answer']/n:.0f}%)  "
                  f"no_hedge_confident={u['no_hedge_confident']}/{n} ({100*u['no_hedge_confident']/n:.0f}%)")
        else:
            print(f"--- Uncertainty Expression: {model} --- FILE NOT FOUND")

        print()
        if r["para"]:
            c = r["para"]
            n = c["n"]
            print(f"--- Consistency: {model} --- (n={n} paraphrase pairs)")
            print(f"  mean cosine similarity={c['mean_cosine_similarity']:.3f}  "
                  f"technique_agreement={c['technique_agreement']}/{n} "
                  f"({100*c['technique_agreement']/n:.0f}%)")
        else:
            print(f"--- Consistency: {model} --- FILE NOT FOUND")


# ---------------------------------------------------------------------------
# HTML rendering -- a single, self-contained, styled report. All CSS is
# embedded in the page itself, so it opens correctly in any browser with
# no internet connection and no other files needed alongside it.
# ---------------------------------------------------------------------------

NAVY = "#14233B"
TEAL = "#0E6E6B"
CORAL = "#C1502E"
LIGHT_TEAL = "#D9EDEA"
LIGHT_GREY = "#F4F7FA"


def esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def num(value, pattern="{:.3f}"):
    return pattern.format(value) if value is not None else "\u2014"


def bar_row(label, val1, val2, n, unit="%"):
    """One labelled row with two horizontal bars, for visually comparing
    two percentage-style values at a glance."""
    pct1 = 100 * val1 / n if n else 0
    pct2 = 100 * val2 / n if n else 0
    return f"""
    <div class="bar-row">
      <div class="bar-label">{esc(label)}</div>
      <div class="bar-track">
        <div class="bar-fill mistral" style="width:{pct1:.1f}%"></div>
        <span class="bar-value">{val1}/{n} ({pct1:.0f}{unit})</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill llama" style="width:{pct2:.1f}%"></div>
        <span class="bar-value">{val2}/{n} ({pct2:.0f}{unit})</span>
      </div>
    </div>"""


def table(headers, rows):
    thead = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{esc(c)}</td>" for c in row)
        trs += f"<tr>{tds}</tr>"
    return f'<table><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>'


def build_html(results):
    models = list(results.keys())
    m1, m2 = results[models[0]], results[models[1]]
    s1, s2 = m1["soc"], m2["soc"]
    a1, a2 = m1["ate"], m2["ate"]
    u1, u2 = m1["adv"], m2["adv"]
    c1, c2 = m1["para"], m2["para"]
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    # --- Faithfulness: SOC scenarios table ---
    soc_table = table(
        ["Metric", models[0], models[1]],
        [
            ["Mean score", num(s1["mean_score"]) if s1 else "\u2014", num(s2["mean_score"]) if s2 else "\u2014"],
            ["Fabricated citations",
             f"{s1['fabricated']}/{s1['n']}" if s1 else "\u2014",
             f"{s2['fabricated']}/{s2['n']}" if s2 else "\u2014"],
            ["Unrecognised-format citations",
             f"{s1['unrecognised']}/{s1['n']}" if s1 else "\u2014",
             f"{s2['unrecognised']}/{s2['n']}" if s2 else "\u2014"],
        ]
    )

    # --- Faithfulness: CTIBench-ATE table ---
    ate_table = table(
        ["Metric", models[0], models[1]],
        [
            ["Precision", num(a1["precision"]) if a1 else "\u2014", num(a2["precision"]) if a2 else "\u2014"],
            ["Recall", num(a1["recall"]) if a1 else "\u2014", num(a2["recall"]) if a2 else "\u2014"],
            ["F1", num(a1["f1"]) if a1 else "\u2014", num(a2["f1"]) if a2 else "\u2014"],
            ["Total fabricated citations",
             str(a1["fabricated_total"]) if a1 else "\u2014",
             str(a2["fabricated_total"]) if a2 else "\u2014"],
        ]
    )

    # --- Uncertainty Expression: bar comparison ---
    uncertainty_bars = ""
    if u1 and u2:
        uncertainty_bars = (
            bar_row("Clean hedge", u1["clean_hedge"], u2["clean_hedge"], u1["n"])
            + bar_row("Hedge then answer", u1["hedge_then_answer"], u2["hedge_then_answer"], u1["n"])
            + bar_row("No hedge, confident", u1["no_hedge_confident"], u2["no_hedge_confident"], u1["n"])
        )
    else:
        uncertainty_bars = '<p class="missing-note">Result file(s) missing for this dimension.</p>'

    # --- Consistency table ---
    consistency_table = table(
        ["Metric", models[0], models[1]],
        [
            ["Mean cosine similarity",
             num(c1["mean_cosine_similarity"]) if c1 else "\u2014",
             num(c2["mean_cosine_similarity"]) if c2 else "\u2014"],
            ["Technique agreement",
             f"{c1['technique_agreement']}/{c1['n']} ({100*c1['technique_agreement']/c1['n']:.0f}%)" if c1 else "\u2014",
             f"{c2['technique_agreement']}/{c2['n']} ({100*c2['technique_agreement']/c2['n']:.0f}%)" if c2 else "\u2014"],
        ]
    )

    missing = [f"{model} \u2014 {dim}" for model, r in results.items()
               for dim, val in r.items() if val is None]
    missing_html = ""
    if missing:
        items = "".join(f"<li>{esc(m)}</li>" for m in missing)
        missing_html = f"""
        <section class="card warning">
          <h2>Missing Result Files</h2>
          <p>The following result files were not found in <code>results/</code>
             when this report was generated:</p>
          <ul>{items}</ul>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TrustSOC Triad &mdash; Phase 4 Evaluation Summary</title>
<style>
  :root {{
    --navy: {NAVY}; --teal: {TEAL}; --coral: {CORAL};
    --light-teal: {LIGHT_TEAL}; --light-grey: {LIGHT_GREY};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #fff; color: #1a1a1a; margin: 0; padding: 0;
  }}
  header {{
    background: var(--navy); color: #fff; padding: 36px 48px;
  }}
  header h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
  header p {{ margin: 0; color: #cfd8e6; font-size: 14px; }}
  main {{ max-width: 900px; margin: 0 auto; padding: 32px 48px 64px; }}
  .card {{
    background: var(--light-grey); border-radius: 10px;
    padding: 24px 28px; margin-bottom: 28px;
  }}
  .card h2 {{
    color: var(--navy); margin: 0 0 14px 0; font-size: 19px;
    border-bottom: 2px solid var(--teal); padding-bottom: 8px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
  th {{
    background: var(--teal); color: #fff; text-align: left;
    padding: 10px 14px; font-weight: 600;
  }}
  th:not(:first-child), td:not(:first-child) {{ text-align: center; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #dde3ea; }}
  tbody tr:nth-child(even) {{ background: var(--light-teal); }}
  .bar-row {{ margin-bottom: 16px; }}
  .bar-label {{ font-weight: 600; color: var(--navy); margin-bottom: 4px; }}
  .bar-track {{
    position: relative; background: #e6ebf0; border-radius: 5px;
    height: 26px; margin-bottom: 4px; overflow: hidden;
  }}
  .bar-fill {{ height: 100%; border-radius: 5px 0 0 5px; }}
  .bar-fill.mistral {{ background: var(--teal); }}
  .bar-fill.llama {{ background: var(--coral); }}
  .bar-value {{
    position: absolute; left: 10px; top: 3px; font-size: 13px;
    font-weight: 600; color: var(--navy); mix-blend-mode: normal;
  }}
  .legend {{ font-size: 13px; color: #555; margin-bottom: 12px; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px;
    margin-right: 6px; vertical-align: middle; }}
  .legend .m {{ background: var(--teal); }}
  .legend .l {{ background: var(--coral); margin-left: 18px; }}
  .missing-note {{ color: #888; font-style: italic; }}
  .warning {{ background: #FBEAE3; border-left: 4px solid var(--coral); }}
  .warning h2 {{ border-bottom-color: var(--coral); }}
  .warning code {{ background: #fff; padding: 1px 5px; border-radius: 3px; }}
  footer {{ text-align: center; color: #999; font-size: 12px; padding: 24px; }}
  @media print {{
    header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    th {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .card {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<header>
  <h1>TrustSOC Triad &mdash; Phase 4 Evaluation Summary</h1>
  <p>Generated {esc(generated)} from result files in <code>results/</code></p>
</header>

<main>

  <section class="card">
    <h2>Faithfulness &mdash; SOC Scenarios</h2>
    {soc_table}
  </section>

  <section class="card">
    <h2>Faithfulness &mdash; CTIBench-ATE</h2>
    {ate_table}
  </section>

  <section class="card">
    <h2>Uncertainty Expression &mdash; Adversarial Probes</h2>
    <div class="legend">
      <span class="m"></span>{esc(models[0])}
      <span class="l"></span>{esc(models[1])}
    </div>
    {uncertainty_bars}
  </section>

  <section class="card">
    <h2>Consistency &mdash; Paraphrase Pairs</h2>
    {consistency_table}
  </section>

  {missing_html}

</main>

<footer>TrustSOC &mdash; automatically generated by build_triad_summary.py</footer>

</body>
</html>"""


def main():
    results = gather_all_results()

    print_console_report(results)

    os.makedirs(FOLDER, exist_ok=True)
    report_path = os.path.join(FOLDER, "triad_summary.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_html(results))

    print(f"\nHTML report written to {report_path}")


if __name__ == "__main__":
    main()
