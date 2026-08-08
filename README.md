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

The system answers SOC-analyst questions using a 35,000+ chunk knowledge
base built from three authoritative sources:

- **MITRE ATT&CK** (v15) — the standard taxonomy of adversary techniques
- **NVD** (National Vulnerability Database) — CVE records, 2023–2025, critical/high severity
- **CISA KEV** (Known Exploited Vulnerabilities) — vulnerabilities confirmed exploited in the wild

Two open-weight models — **Mistral 7B** and **LLaMA 3 8B**, both run
locally via [Ollama](https://ollama.com) — are compared with and without
retrieval, on the same questions, to measure faithfulness, consistency,
and uncertainty expression (the **TrustSOC Triad**).

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
├── pipeline/                core code - parsers, knowledge base builder, RAG pipeline
│   ├── parse_attack.py          parses the MITRE ATT&CK STIX bundle
│   ├── parse_vulnerabilities.py parses NVD CVE feeds and the CISA KEV CSV
│   ├── build_kb.py              embeds and indexes everything into ChromaDB
│   └── rag_pipeline.py          the citation-enforcing prompt, retriever, and generation
├── ui/
│   └── dashboard.py         Streamlit dashboard - visual demo with a trust-signals panel
├── demo/
│   ├── Live_Demo.py             interactive terminal demo (RAG vs no-RAG, side by side)
│   └── test_retrieval.py        standalone retrieval-quality check
├── kb/                      generated ChromaDB vector store (not committed - see below)
├── evaluation/, results/, figures/, tests/   Phase 4/5 evaluation work (in progress)
├── check_setup.py           verifies the environment is set up correctly
├── setup_project.py         one-time script that scaffolded this project (already run)
├── requirements.txt         pinned dependency versions
└── .gitignore
```

---

## How it works

1. **Parsing** (`pipeline/parse_attack.py`, `pipeline/parse_vulnerabilities.py`) -
   each source is turned into small, self-contained text chunks tagged with
   metadata (`source`, `technique_id`/`cve_id`, etc.).
2. **Indexing** (`pipeline/build_kb.py`) - every chunk is embedded with
   `BAAI/bge-small-en-v1.5` and stored in a persistent ChromaDB collection.
   Re-running this script is safe - it only indexes chunks that aren't
   already present.
3. **Retrieval** (`pipeline/rag_pipeline.py`, function `retrieve()`) - for
   a given question, the top-k nearest chunks are retrieved. Retrieval is
   **source-balanced**: half the results are guaranteed to come from
   MITRE ATT&CK and half from NVD/KEV, so a query about a technique isn't
   drowned out by the much larger number of CVE records (32,779 CVEs vs
   697 ATT&CK techniques in this knowledge base).
4. **Generation** (`pipeline/rag_pipeline.py`, function `trustsoc_answer()`) -
   the retrieved chunks are inserted into a citation-enforcing prompt and
   sent to the chosen model via Ollama at `temperature=0`. The prompt
   requires the model to cite its sources in `[MITRE ATT&CK Txxxx]` format
   and to explicitly say when it doesn't have enough information, rather
   than guess.

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

**CTIBench-ATE**, used as an independent evaluation benchmark in Phase 4,
is deliberately **not** part of the knowledge base (to avoid data
leakage - see "Design notes" below) and is not yet included in this repo.
It will be added under `benchmarks/cti-bench/` once Phase 4 evaluation
begins.

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
python demo\test_retrieval.py
```
Runs three representative queries and prints the top matches with their
distances - useful after rebuilding the knowledge base, or after changing
any of the data sources.

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

---

## Design notes

- **RAGAS was evaluated and removed.** The original design planned to
  measure faithfulness with both RAGAS and a custom ATT&CK
  technique-alignment scorer. RAGAS caused unresolvable dependency
  conflicts with current LangChain, and - independently - is arguably
  less suited to a domain where correctness is objectively checkable
  (a technique ID either matches the ground truth or it doesn't) rather
  than requiring an LLM's subjective judgement. Faithfulness is measured
  using the custom scorer alone.
- **CTIBench is excluded from the knowledge base** even though the wider
  project uses it, specifically to prevent the system from retrieving
  the benchmark's own answers and inflating its evaluation score.
- **Retrieval is source-balanced**, not naive top-k similarity search,
  because naive search let the much larger number of CVE chunks
  (32,779) drown out relevant ATT&CK techniques (697) for
  technique-style queries. See `pipeline/rag_pipeline.py`, `retrieve()`.

---

## Project status

| Phase | Status |
|---|---|
| 0 - Environment setup | Complete |
| 1 - Literature review | In progress |
| 2 - Knowledge base | Complete (35,111 chunks; retrieval tested and fixed) |
| 3 - RAG pipeline & dashboard | Complete (both models tested, terminal + dashboard demos working) |
| 4 - Evaluation datasets | In progress |
| 5 - Results & statistical analysis | Not started |

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError` on any script | The virtual environment isn't active - run `trustsoc_env\Scripts\activate` first |
| `ModuleNotFoundError: No module named 'pipeline'` when running from `demo/` or `ui/` | Run scripts from the project root (`trustsoc\`), not from inside `demo/` or `ui/` |
| Ollama connection errors | Make sure the Ollama app is running (check the system tray), or run `ollama serve` |
| `chromadb.errors.InternalError: too many SQL variables` | Fixed in `build_kb.py` - existing IDs are now fetched in paginated batches |
| Dashboard crashes with a CUDA/stack-overrun error | VRAM exhausted - reduce "Sources to retrieve" to 5, especially when using `llama3` |
