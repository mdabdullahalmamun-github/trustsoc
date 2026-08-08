"""Builds the TrustSOC knowledge base: parses MITRE ATT&CK, NVD, and CISA KEV,
embeds every chunk with BGE-small-en, and writes them into a persistent
ChromaDB collection at kb/. Safe to re-run — existing chunk IDs are skipped,
so it only indexes what's new. CTIBench is deliberately excluded here (see
README) to avoid data leakage on the CTIBench-ATE evaluation in Phase 4."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))  # allow imports from this folder

import chromadb
from chromadb.utils import embedding_functions

from parse_attack import load_attack
from parse_vulnerabilities import load_nvd, load_kev

# Embedding model used for every chunk (BGE-small, recommended for retrieval tasks)
bge = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="kb")
collection = client.get_or_create_collection(
    name="cti",
    embedding_function=bge,
    metadata={"hnsw:space": "cosine"},  # cosine similarity, set explicitly rather than left as default
)

# Load and combine all three sources
attack_chunks = load_attack()
nvd_chunks = load_nvd(paths=("data/nvd_2023.json", "data/nvd_2024.json"))
kev_chunks = load_kev()
all_chunks = attack_chunks + nvd_chunks + kev_chunks
print(f"Total chunks to index: {len(all_chunks)}")

# Idempotent indexing: skip any chunk ID that's already in the collection,
# so re-running this script only adds what's new instead of duplicating everything
# Fetch existing IDs in batches — a single unbatched get() on a large collection
# can exceed SQLite's internal limit on query parameters ("too many SQL variables").
existing = set()
if collection.count():
    offset = 0
    page_size = 5000
    while True:
        page = collection.get(limit=page_size, offset=offset, include=[])["ids"]
        if not page:
            break
        existing.update(page)
        offset += page_size
BATCH_SIZE = 256
batch = []

def flush(items):
    """Write a batch of chunks to ChromaDB."""
    if not items:
        return
    collection.add(
        ids=[x["id"] for x in items],
        documents=[x["text"] for x in items],
        metadatas=[x["metadata"] for x in items],
    )

for chunk in all_chunks:
    if chunk["id"] in existing:
        continue
    batch.append(chunk)
    if len(batch) >= BATCH_SIZE:
        flush(batch)
        batch = []
flush(batch)

print(f"Knowledge base now holds {collection.count()} chunks.")