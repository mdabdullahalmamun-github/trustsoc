"""
One-time cleanup: removes just the MITRE ATT&CK chunks from the knowledge
base (IDs starting with "attack-"), leaving all NVD and CISA KEV chunks
untouched. Needed because build_kb.py's idempotent indexing skips any ID
that already exists -- so after patching parse_attack.py to strip inline
(Citation: X) markers, simply re-running build_kb.py would NOT actually
re-embed the cleaned text, since the same 697 attack-* IDs are already
present. Run this once, then run build_kb.py to re-add them with the
cleaned text.

Run from your project root:
    python pipeline/clear_attack_chunks.py
"""
import chromadb

client = chromadb.PersistentClient(path="kb")
collection = client.get_collection(name="cti")

before_total = collection.count()

# Fetch all IDs in batches (same "too many SQL variables" limit as
# build_kb.py's existing-ID check applies here too on a large collection).
all_ids = []
offset = 0
page_size = 5000
while True:
    page = collection.get(limit=page_size, offset=offset, include=[])["ids"]
    if not page:
        break
    all_ids.extend(page)
    offset += page_size

attack_ids = [i for i in all_ids if i.startswith("attack-")]
print(f"Knowledge base currently holds {before_total} chunks.")
print(f"Found {len(attack_ids)} MITRE ATT&CK chunks to remove.")

if not attack_ids:
    print("Nothing to do -- no attack-* IDs found.")
else:
    # Delete in batches too, for the same reason
    for i in range(0, len(attack_ids), 5000):
        collection.delete(ids=attack_ids[i:i + 5000])
    after_total = collection.count()
    print(f"Removed {before_total - after_total} chunks.")
    print(f"Knowledge base now holds {after_total} chunks "
          f"(should be exactly {before_total - len(attack_ids)} -- NVD/KEV untouched).")
    print("\nNow run: python pipeline/build_kb.py")
    print("to re-add the ATT&CK chunks with citation markers stripped.")
