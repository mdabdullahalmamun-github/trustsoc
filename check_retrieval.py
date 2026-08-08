"""Manual retrieval-quality check (not an automated test — see tests/ for those
once Phase 4 adds them). Re-run any time after rebuilding the knowledge base or
changing pipeline/rag_pipeline.py to confirm all three query types return
sensible results. Run from the project root: python check_retrieval.py"""
from pipeline.rag_pipeline import retrieve, collection

print("Chunks:", collection.count())
for q in ["lateral movement using remote services",
          "PowerShell encoded command execution",
          "actively exploited vulnerability in a VPN"]:
    print(f"\nQUERY: {q}")
    for hit in retrieve(q, 6):
        print(f"  [{hit['distance']}] {hit['source']:14} {hit['id']}: {hit['text'][:70]}...")
