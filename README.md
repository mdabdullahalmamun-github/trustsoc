# TrustSOC

**A Multi-Dataset RAG Framework for Faithful Cyber Threat Analysis**

MSc Cybersecurity dissertation project — Md Abdullah Al Mamun (2010022),
University of Bedfordshire. Supervisor: Dr Monika Roopak.

---

## What this is

Large Language Models are increasingly used to help Security Operations
Centre (SOC) analysts triage alerts and research threats — but an
ungrounded model can hallucinate confidently and plausibly, which is
dangerous in a security context. TrustSOC investigates whether
Retrieval-Augmented Generation (RAG) — grounding an LLM's answers in
retrieved, authoritative documents — produces measurably more trustworthy
outputs than the same model with no grounding at all.

The system answers SOC-analyst questions using a 35,111-chunk knowledge
base built from three authoritative sources:

- **MITRE ATT&CK** (v15) — the standard taxonomy of adversary techniques
- **NVD** (National Vulnerability Database) — CVE records, 2023–2025, critical/high severity
- **CISA KEV** (Known Exploited Vulnerabilities) — vulnerabilities confirmed exploited in the wild

Two open-weight models — **Mistral 7B** and **LLaMA 3 8B**, both run
locally via [Ollama](https://ollama.com) — were compared with and without
retrieval, on identical questions across four separate test sets, to measure
faithfulness, consistency, and uncertainty expression (the **TrustSOC
Triad**). Full evaluation results are summarised below.

### Headline finding

No single model was more trustworthy overall — but the most striking result
concerns *why* a model appears cautious at all. On 30 adversarial probes
designed to tempt fabrication, LLaMA 3 correctly hedged on **30/30** when
grounded with retrieval. With retrieval switched off, that fell to **0/30**:
the model confidently invented a fictitious ransomware campaign and
attributed it to a real named threat-actor group. LLaMA 3's apparent caution
turned out to be a property of the RAG pipeline itself, not an inherent trait
of the model. See [Results](#results) below for the full picture, including
where Mistral performed better.

### A concrete example of the problem this solves

Asked "In MITRE ATT&CK, what is technique T1059 and which tactic does
it belong to?" without any grounding, the model confidently answers
incorrectly. With RAG, grounded in the real ATT&CK entry, it answers
correctly and cites its source:

```
--- WITHOUT RAG ---
Technique T1059 ... known as "File and Directory Discovery." It belongs
to the Tactic of Initial Access.                                    [wrong]

--- WITH RAG ---
Technique T1059 belongs to the tactic of execution [MITRE ATT&CK]. It
involves the abuse of command and script interpreters to execute
commands, scripts, or binaries.                                     [correct, cited]
```

---

## Project structure

```
trustsoc/
├── data/                    raw source data (ATT&CK, NVD, KEV)
├── benchmarks/
│   └── cti-bench/           CTIBench-ATE (Enterprise-only subset), used as an
│                             independent faithfulness benchmark - kept OUT of
│                             the knowledge base to avoid data leakage
├── pipeline/                core code - parsers, knowledge base builder, RAG pipeline
│   ├── parse_attack.py          parses the MITRE ATT&CK STIX bundle
│   ├── parse_vulnerabilities.py parses NVD CVE feeds and the CISA KEV CSV
│   ├── build_kb.py              embeds and indexes everything into ChromaDB
│   ├── build_ctibench_ate.py    builds the 47-item Enterprise-only CTIBench-ATE test set
│   ├── clear_attack_chunks.py   utility - clears ATT&CK chunks from the KB for a clean rebuild
│   ├── verify_citation_strip.py one-off check confirming MITRE's own inline citation
│   │                             markers were fully stripped from ATT&CK chunks (see
│   │                             "Design notes" - this was a real bug found during Phase 4)
│   └── rag_pipeline.py          the citation-enforcing prompt, retriever, citation
│                                 validation, and generation - trustsoc_answer()
├── evaluation/               Phase 4 evaluation harness
│   ├── soc_scenarios.json           50 hand-written SOC-analyst questions (test-set definition)
│   ├── ctibench_ate.json            the 47-item CTIBench-ATE test set (test-set definition)
│   ├── adversarial_probes.json      30 probes: fabricated IOCs, future-dated threats,
│   │                                 unknown actor names (test-set definition)
│   ├── paraphrase_pairs.json        50 reworded-question pairs (test-set definition)
│   ├── run_soc_scenarios.py         runs a test set through trustsoc_answer() and scores it
│   ├── run_ctibench_ate.py            (one run_*.py per test set; each has a matching
│   ├── run_adversarial_probes.py       _baseline variant that runs with RAG disabled,
│   ├── run_paraphrase_pairs.py         for the RAG-vs-no-RAG comparison)
│   └── build_triad_summary.py       aggregates every result file in results/ into one
│                                     console summary + a styled HTML report
├── results/                  Phase 4 output - every *_results_*.json file, plus
│                              triad_summary.html (open directly in a browser)
├── figures/
│   ├── visualise_kb_full.py     generates an interactive visualisation of the knowledge base
│   └── kb_full_interactive.html   its output - a zoomable UMAP projection of all 35,111 chunks
├── ui/
│   └── dashboard.py          Streamlit dashboard - visual demo with a trust-signals panel
├── demo/
│   └── Live_Demo.py              interactive terminal demo (RAG vs no-RAG, side by side)
├── kb/                        generated ChromaDB vector store (not committed - see below)
├── tests/                     reserved for automated unit tests (not yet populated -
│                               see "Project status")
├── check_setup.py             verifies the environment is set up correctly
├── check_retrieval.py         manual retrieval-quality check (run after building the knowledge base)
├── setup_project.py           one-time script that scaffolded this project (already run)
├── requirements.txt           pinned dependency versions
└── .gitignore
```

> `check_retrieval.py` was moved from `demo/` to the project root and renamed
> from `test_retrieval.py` — it's a manual, human-read sanity check (not an
> automated test, despite the old name), so it's grouped with `check_setup.py`
> rather than the interactive demos or the `tests/` folder.

> **Note on `evaluation/` vs `results/` vs `tests/`:** these look similar but
> serve different purposes. `evaluation/` holds the harness — the code that
> runs a test set and the test-set definitions themselves (inputs).
> `results/` holds what that code *produced* (outputs) — kept separate so
> it's always obvious which files are safe to regenerate by re-running a
> script versus which are the harness itself. `tests/` is reserved for
> conventional unit tests that check code correctness (e.g. "does
> `parse_attack()` extract the right technique ID from this fixture") — a
> different job from evaluating model behaviour, which is what everything
> in `evaluation/` does.

---

## How it works

1. **Parsing** (`pipeline/parse_attack.py`, `pipeline/parse_vulnerabilities.py`) -
   each source is turned into small, self-contained text chunks tagged with
   metadata (`source`, `technique_id`/`cve_id`, etc.). MITRE's own inline
   citation markers (`(Citation: ...)`) are stripped from the ATT&CK text
   during parsing — see "Design notes" for why this matters.
2. **Indexing** (`pipeline/build_kb.py`) - every chunk is embedded with
   `BAAI/bge-small-en-v1.5` and stored in a persistent ChromaDB collection.
   Re-running this script is safe - it only indexes chunks that aren't
   already present.
3. **Retrieval** (`pipeline/rag_pipeline.py`, function `retrieve()`) - for
   a given question, the top-k nearest chunks are retrieved. Retrieval is
   **source-balanced**: half the results are guaranteed to come from
   MITRE ATT&CK and half from NVD/KEV, so a query about a technique isn't
   drowned out by the much larger number of CVE records (32,779 CVEs vs
   697 ATT&CK techniques in this knowledge base — a 47:1 imbalance).
   Retrieval is also **relevance-filtered**: chunks beyond a cosine-distance
   threshold (`RELEVANCE_THRESHOLD`, currently 0.5) are discarded rather
   than handed to the model regardless of quality. If nothing passes the
   threshold, the system returns the standard insufficient-context message
   without calling the model at all. This threshold is a reasoned starting
   point rather than an empirically calibrated one — see "Project status"
   and "Design notes".
4. **Generation** (`pipeline/rag_pipeline.py`, function `trustsoc_answer()`) -
   the retrieved chunks are inserted into a citation-enforcing prompt and
   sent to the chosen model via Ollama at `temperature=0`. The prompt
   requires the model to cite its sources in `[MITRE ATT&CK Txxxx]` format
   and to explicitly say when it doesn't have enough information, rather
   than guess.
5. **Citation validation** (`pipeline/rag_pipeline.py`, function
   `validate_citations()`) - every technique ID and CVE ID the model cites
   in its answer is checked against what was actually retrieved, flagging
   fabricated citations (IDs that were never retrieved) separately from
   unrecognised-format citations (real-looking references that don't match
   the required `[Source ID]` format). This feeds directly into the Phase 4
   faithfulness scoring in `evaluation/`.

---

## Prerequisites

- **Python 3.11 or newer** (this project was built and tested on 3.12)
- **[Ollama](https://ollama.com)**, installed and running
- **~10 GB free disk space** for the two models, plus space for the raw
  data and knowledge base
- Tested on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM); a full-precision
  7-8B model query took ~9 seconds on this hardware

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv trustsoc_env
trustsoc_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the two models (a few GB each, first time only)
ollama pull mistral
ollama pull llama3

# 4. Verify everything is working
python check_setup.py
```

`check_setup.py` confirms every required library imports correctly and
that Ollama is reachable with both models present.

---

## Getting the data

Three of the four data sources are too large to include in this
repository (500+ MB combined) and are excluded via `.gitignore`.
Download them manually into `data/` before building the knowledge base:

| Source | Where to get it | Save as |
|---|---|---|
| MITRE ATT&CK v15 (STIX) | github.com/mitre-attack/attack-stix-data -> `enterprise-attack/enterprise-attack.json` | `data/enterprise-attack.json` |
| NVD CVE feeds (API 2.0) | nvd.nist.gov/developers/vulnerabilities | `data/nvd_2023.json`, `data/nvd_2024.json` |
| CISA KEV catalogue (CSV) | cisa.gov/known-exploited-vulnerabilities-catalog | `data/cisa_kev.csv` |

**CTIBench-ATE**, used as an independent evaluation benchmark, is
deliberately **not** part of the knowledge base (to avoid data leakage —
see "Design notes" below). It lives under `benchmarks/cti-bench/`; the
47-item Enterprise-only subset used in evaluation is built from it by
`pipeline/build_ctibench_ate.py` into `evaluation/ctibench_ate.json`
(13 Mobile-platform items are excluded, since the knowledge base has no
Mobile ATT&CK coverage to ground them against).

---

## Building the knowledge base

```powershell
python pipeline\build_kb.py
```

Expected output (first run):
```
Parsed 697 ATT&CK techniques
NVD CVEs: 32779   CISA KEV: 1635
Total chunks to index: 35111
Knowledge base now holds 35111 chunks.
```

This is safe to re-run at any time - existing chunks are skipped, so it
only indexes anything new.

**Quick sanity check that retrieval actually works:**
```powershell
python check_retrieval.py
```
Runs three representative queries and prints the top matches with their
distances - useful after rebuilding the knowledge base, or after changing
`pipeline/rag_pipeline.py`.

**If you ever rebuild the ATT&CK chunks specifically** (e.g. via
`pipeline/clear_attack_chunks.py` followed by a rebuild), it's worth
re-running `pipeline/verify_citation_strip.py` afterwards to confirm no
inline `(Citation: ...)` markers made it back into the chunk text — see
"Design notes" for why this matters.

---

## Running the demo

**Terminal, interactive, side-by-side comparison:**
```powershell
python demo\Live_Demo.py
```
Type any SOC-analyst question; see the answer with and without RAG,
back to back. Type `quit` to exit.

**Visual dashboard:**
```powershell
streamlit run ui\dashboard.py
```
Opens in your browser. Lets you pick the model, toggle RAG on/off, adjust
how many sources are retrieved, and see a live trust-signals panel
(Grounded? / Citations / Hedged?) alongside the answer and its sources.

> **Known hardware limit:** on a 4 GB GPU, running `llama3` with the
> "Sources to retrieve" slider set to 10 has crashed the local Ollama
> backend (VRAM exhaustion). Keep `k=5` for reliable results, especially
> with the larger model.

> **Note on the "Sources to retrieve" slider:** this sets a maximum, not a
> guaranteed count. If fewer chunks pass the relevance threshold than the
> slider requests, fewer are shown — the system never pads results with
> irrelevant chunks just to reach the selected number.

---

## Running the evaluation

Each test set has its own runner in `evaluation/`, plus a `_baseline`
variant that runs the same questions with retrieval switched off (used for
the RAG-vs-no-RAG comparison in the Results section below). All of them
write their per-item output to `results/`:

```powershell
python evaluation\run_soc_scenarios.py
python evaluation\run_ctibench_ate.py
python evaluation\run_adversarial_probes.py
python evaluation\run_paraphrase_pairs.py
# each also has a _baseline.py / _mistral.py / _llama3.py variant -
# see evaluation/ for the full set
```

Once the result files you need exist in `results/`, generate the combined
summary:

```powershell
python evaluation\build_triad_summary.py
```

This prints a console summary and writes `results/triad_summary.html` — a
self-contained, styled report comparing both models side by side across
all three Triad dimensions, with visual bar charts for the uncertainty
comparison. Open it directly in a browser; no other files or an internet
connection are needed. Safe to re-run any time an underlying result file
changes — it always reflects whatever is currently in `results/`, and
clearly reports any expected file that's missing rather than failing
silently.

---

## Results

Full statistical analysis (Wilcoxon signed-rank as the primary test, since
the design is paired; Mann-Whitney U reported alongside for comparison) is
in the dissertation. The headline numbers:

**Faithfulness**

| Metric | Mistral 7B | LLaMA 3 8B |
|---|---|---|
| SOC scenarios — mean score (n=50) | 0.420 | 0.360 |
| SOC scenarios — fabricated citations | 14/50 | 6/50 |
| CTIBench-ATE — F1 (n=47) | 0.234 | 0.157 |
| CTIBench-ATE — fabricated citations (total) | 246 | 45 |

**Consistency** (50 paraphrase pairs)

| Metric | Mistral 7B | LLaMA 3 8B |
|---|---|---|
| Mean cosine similarity | 0.858 | 0.785 |
| Technique agreement | 32/50 (64%) | 25/50 (50%) |

**Uncertainty Expression** (30 adversarial probes, manually verified)

| Condition | Mistral 7B | LLaMA 3 8B |
|---|---|---|
| Genuine caution — WITH retrieval | 2/30 (7%) | 30/30 (100%) |
| Genuine caution — WITHOUT retrieval | 3/30 (10%) | 0/30 (0%) |

**RAG vs no-RAG** (SOC scenarios, mean score)

| Model | With RAG | Without RAG |
|---|---|---|
| Mistral 7B | 0.470 | 0.000 |
| LLaMA 3 8B | 0.300 | 0.020 |

**In short:** Mistral was more faithful and more consistent, but
fabricated substantially more often. LLaMA 3 showed near-perfect caution —
but only when grounded; that caution collapsed entirely without retrieval,
including inventing a fictitious ransomware campaign attributed to a real
threat-actor group. Neither model was straightforwardly "more trustworthy"
overall — the two dimensions pulled in different directions, which is
itself the project's central finding.

---

## Design notes

- **A real bug found during Phase 4: MITRE's own citation markers were
  leaking into the knowledge base.** ATT&CK's source STIX data contains
  inline references like `(Citation: Some Vendor Report)`, embedded
  directly in the technique description text. These were being indexed
  as-is, so models sometimes echoed them back verbatim — which
  `validate_citations()` then correctly flagged as unrecognised-format
  citations, but for the wrong reason (a leaked source artefact, not a
  genuine fabrication). This was only visible at real evaluation volume;
  weeks of manual testing hadn't surfaced it. Fixed by stripping citation
  markers from all 697 ATT&CK chunks during parsing and rebuilding the
  knowledge base; `pipeline/verify_citation_strip.py` confirms none remain.
- **RAGAS was evaluated and removed.** The original design planned to
  measure faithfulness with both RAGAS and a custom ATT&CK
  technique-alignment scorer. RAGAS caused unresolvable dependency
  conflicts with current LangChain, and - independently - is arguably
  less suited to a domain where correctness is objectively checkable
  (a technique ID either matches the ground truth or it doesn't) rather
  than requiring an LLM's subjective judgement. Faithfulness is measured
  using the custom scorer alone.
- **CTIBench-ATE is excluded from the knowledge base** even though the
  wider project uses it, specifically to prevent the system from
  retrieving the benchmark's own answers and inflating its evaluation
  score.
- **Retrieval is source-balanced**, not naive top-k similarity search,
  because naive search let the much larger number of CVE chunks
  (32,779) drown out relevant ATT&CK techniques (697) for
  technique-style queries. See `pipeline/rag_pipeline.py`, `retrieve()`.
- **Retrieval is relevance-filtered**, not just top-k. Earlier, `retrieve()`
  always returned exactly `k` chunks regardless of how relevant they
  actually were, so the model's own refusal instruction was the only
  safeguard against being handed poor-quality context. A cosine-distance
  threshold (`RELEVANCE_THRESHOLD = 0.5`) now makes retrieval itself
  responsible for recognising when nothing useful was found. This value
  remains a reasoned starting point rather than an empirically calibrated
  one — calibrating it against hand-labelled examples is documented as
  future work rather than completed in this project.
- **Citations are validated, not just requested.** The prompt asks the
  model to cite technique/CVE IDs, but nothing previously checked whether
  those IDs were genuinely present in the retrieved context.
  `validate_citations()` closes that gap, separating genuinely fabricated
  citations from unrecognised-format ones.
- **Wilcoxon signed-rank, not Mann-Whitney U, is the primary significance
  test.** The evaluation design is paired — the same question is asked of
  the same model with and without retrieval — so a paired test is the
  statistically correct choice. This mattered in practice, not just in
  theory: on the SOC-scenario RAG-vs-no-RAG comparison, Mann-Whitney gave
  a non-significant result (p=0.067) while the correct paired test gave a
  clearly significant one (p=0.0035). Both are reported rather than only
  the more favourable number.
- **Generation failures are non-fatal.** `generate()` catches connection
  errors from Ollama (not just malformed responses) and retries, so a
  single dropped connection during a long batch evaluation run degrades
  to one logged failure rather than crashing the entire run.

---

## Project status

| Phase | Status |
|---|---|
| 0 - Environment setup | Complete |
| 1 - Literature review | Complete |
| 2 - Knowledge base | Complete (35,111 chunks; retrieval tested and fixed) |
| 3 - RAG pipeline & dashboard | Complete (both models tested, terminal + dashboard demos working; relevance filtering and citation validation added) |
| 4 - Evaluation datasets & runs | Complete (4 test sets, both models, with and without retrieval; citation-marker bug found and fixed) |
| 5 - Results & statistical analysis | Complete (Wilcoxon/Mann-Whitney comparison, full Triad summary — see Results above) |

**Known limitations, carried forward as future work rather than resolved
here:** `RELEVANCE_THRESHOLD` is a reasoned starting point, not empirically
calibrated against hand-labelled examples; the hedge classifier is
manually verified rather than an independently validated automated
measure; and `tests/` does not yet contain automated unit tests (the
evaluation harness in `evaluation/` tests model behaviour, which is a
different job from testing code correctness).

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError` on any script | The virtual environment isn't active - run `trustsoc_env\Scripts\activate` first |
| `ModuleNotFoundError: No module named 'pipeline'` when running from `demo/`, `ui/`, or `evaluation/` | Run scripts from the project root (`trustsoc\`), not from inside those folders |
| Ollama connection errors | Make sure the Ollama app is running (check the system tray), or run `ollama serve` |
| `chromadb.errors.InternalError: too many SQL variables` | Fixed in `build_kb.py` - existing IDs are now fetched in paginated batches |
| Dashboard crashes with a CUDA/stack-overrun error | VRAM exhausted - reduce "Sources to retrieve" to 5, especially when using `llama3` |
| ChromaDB error requesting 0 results | Fixed in `rag_pipeline.py` - `retrieve()` now guards against zero-count queries at `k=1` |
| `build_triad_summary.py` reports a result file as missing | Run the matching `evaluation/run_*.py` script first - it needs to have written its output to `results/` |
