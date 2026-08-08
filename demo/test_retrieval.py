"""Standalone retrieval-quality check. Re-run any time after rebuilding
the knowledge base to confirm all three query types return sensible results."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import retrieve, collection

print("Chunks:", collection.count())
for q in ["lateral movement using remote services",
          "PowerShell encoded command execution",
          "actively exploited vulnerability in a VPN"]:
    print(f"\nQUERY: {q}")
    for hit in retrieve(q, 6):
        print(f"  [{hit['distance']}] {hit['source']:14} {hit['id']}: {hit['text'][:70]}...")