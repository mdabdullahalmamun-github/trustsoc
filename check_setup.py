import importlib, sys, requests

print(f"Python: {sys.version.split()[0]}  (need 3.11+)\n")

# ragas deliberately not checked — removed due to a LangChain version conflict;
# faithfulness is measured with the custom ATT&CK alignment scorer instead.
libs = ["langchain", "chromadb", "sentence_transformers",
        "datasets", "scipy", "numpy", "pandas", "matplotlib",
        "seaborn", "streamlit", "requests", "tqdm"]
for name in libs:
    try:
        importlib.import_module(name)
        print(f"  \u2713 {name}")
    except Exception as e:
        print(f"  \u2717 {name}  ->  {e}")

# Is the Ollama server running and are both models present?
print("\nOllama check:")
try:
    tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
    have = {m["name"].split(":")[0] for m in tags.get("models", [])}
    for needed in ["mistral", "llama3"]:
        mark = "\u2713" if needed in have else "\u2717 (run:  ollama pull " + needed + ")"
        print(f"  {mark} {needed}")
except Exception as e:
    print(f"  \u2717 Ollama not reachable -> {e}\n  Start it: open the Ollama app, or run 'ollama serve' in a terminal.")