# -*- coding: utf-8 -*-
"""
Builds the CTIBench-ATE evaluation dataset for TrustSOC Phase 4, from the
real CTIBench data (Alam et al., 2024) at github.com/xashru/cti-bench.

Two deliberate departures from CTIBench's own original benchmark protocol,
both justified in the chat discussion and worth restating in Chapter 3/5:

1. ENTERPRISE-ONLY FILTER. The real cti-ate.tsv has 60 rows: 47 Enterprise,
   13 Mobile. TrustSOC's knowledge base only indexes MITRE ATT&CK Enterprise
   v15 -- it never ingested Mobile ATT&CK at all. Testing against the 13
   Mobile rows would be measuring a gap that was never in scope (the system
   has zero chance of retrieving content it was never given), not a real
   weakness. Excluded here, explicitly, rather than silently.

2. TRUSTSOC'S OWN PROMPT, NOT CTIBENCH'S. The original CTIBench prompt
   embeds the full list of ~200+ valid technique names inline, so the model
   being tested picks from a supplied reference list. TrustSOC is being
   tested as a RAG system here -- the whole point is whether its own
   retrieval can find the right techniques without being handed the answer
   space directly. Each row's "Description" (the real MITRE software-page
   text) is reframed as an analyst-style question instead, run through the
   real trustsoc_answer() pipeline exactly like every other test set.

Ground truth remains CTIBench's real, published multi-label answer set
(multiple techniques per item) -- this is NOT single-answer scoring like
the SOC scenarios. Multi-label precision/recall/F1 scoring lives in
run_ctibench_ate.py, matching how the original CTIBench paper itself
evaluates this task.
"""
import csv
import json

SOURCE_TSV = "benchmarks/cti-bench/data/cti-ate.tsv"

with open(SOURCE_TSV, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    rows = list(reader)

print(f"Raw CTIBench-ATE rows: {len(rows)}")

enterprise_rows = [r for r in rows if r[1] == "Enterprise"]
mobile_rows = [r for r in rows if r[1] == "Mobile"]
print(f"Enterprise: {len(enterprise_rows)}  Mobile (excluded): {len(mobile_rows)}")

items = []
for i, row in enumerate(enterprise_rows, 1):
    url, platform, description, prompt, gt = row
    software_id = url.rstrip("/").split("/")[-1]  # e.g. "S0066"
    gt_techniques = sorted(set(t.strip() for t in gt.split(",") if t.strip()))

    question = (
        f"An analyst is investigating a piece of software with the following "
        f"description. Which MITRE ATT&CK techniques does it use?\n\n{description}"
    )

    items.append({
        "id": f"ATE-{i:03d}",
        "mitre_software_id": software_id,
        "source_url": url,
        "question": question,
        "ground_truth_techniques": gt_techniques,
    })

print(f"\nBuilt {len(items)} Enterprise-only evaluation items.")

gt_counts = [len(it["ground_truth_techniques"]) for it in items]
print(f"Ground-truth technique count per item: min={min(gt_counts)}, "
      f"max={max(gt_counts)}, avg={sum(gt_counts)/len(gt_counts):.1f}")

with open("evaluation/ctibench_ate.json", "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

print("\nWrote evaluation/ctibench_ate.json")
print("\nSample item:")
print(json.dumps(items[0], indent=2)[:600])
