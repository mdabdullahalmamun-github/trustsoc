# TrustSOC

A Multi-Dataset RAG Framework for Faithful Cyber Threat Analysis.

MSc Cybersecurity dissertation project — Md Abdullah Al Mamun (2010022),
University of Bedfordshire. Supervisor: Dr Monika Roopak.

## What this is

TrustSOC compares a grounded (RAG) and ungrounded LLM assistant for
Security Operations Centre (SOC) analysts, answering questions using a
35,000+ chunk knowledge base built from MITRE ATT&CK, the National
Vulnerability Database (NVD), and the CISA Known Exploited Vulnerabilities
(KEV) catalogue.

## Project structure

- `data/` — raw source data (ATT&CK, NVD, KEV, CTIBench)
- `pipeline/` — core code: parsers, knowledge base builder, RAG pipeline
- `ui/` — Streamlit dashboard
- `demo/` — interactive terminal demo and retrieval-quality check
- `kb/` — generated ChromaDB vector store (not committed; rebuild with `pipeline/build_kb.py`)
- `evaluation/`, `results/`, `figures/`, `tests/` — Phase 4/5 evaluation outputs (in progress)
- `check_setup.py` — verifies the environment is set up correctly

## Setup

```powershell
python -m venv trustsoc_env
trustsoc_env\Scripts\activate
pip install -r requirements.txt
ollama pull mistral
ollama pull llama3
python check_setup.py
```

## Building the knowledge base

```powershell
python pipeline\build_kb.py
```

## Running the demo

```powershell
python demo\Live_Demo.py
```
or, for the dashboard:
```powershell
streamlit run ui\dashboard.py
```

## Checking retrieval quality

```powershell
python demo\test_retrieval.py
```