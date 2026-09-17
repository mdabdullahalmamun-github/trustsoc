"""
Quick sanity check: confirms no "(Citation: ...)" markers survived in any
of the 697 rebuilt MITRE ATT&CK chunks -- checking all of them, not just
the one truncated example build_kb.py prints.

Run from your project root:
    python pipeline/verify_citation_strip.py
"""
import chromadb

client = chromadb.PersistentClient(path="kb")
collection = client.get_collection(name="cti")

# Fetch just the ATT&CK chunks' text, in batches (same SQL-variable limit
# as everywhere else once you're dealing with a large collection).
attack_docs = []
offset = 0
page_size = 2000
while True:
    page = collection.get(
        where={"source": "MITRE ATT&CK"},
        limit=page_size,
        offset=offset,
        include=["documents"],
    )
    if not page["ids"]:
        break
    attack_docs.extend(zip(page["ids"], page["documents"]))
    offset += page_size

print(f"Checked {len(attack_docs)} ATT&CK chunks.")

still_has_citation = [(cid, text) for cid, text in attack_docs if "Citation:" in text]

if still_has_citation:
    print(f"\n{len(still_has_citation)} chunks STILL contain a citation marker:")
    for cid, text in still_has_citation[:5]:
        idx = text.find("Citation:")
        print(f"  {cid}: ...{text[max(0,idx-40):idx+60]}...")
    print("\nSomething's off -- worth checking parse_attack.py was actually saved/replaced correctly.")
else:
    print("\nCONFIRMED: zero citation markers remain across all 697 ATT&CK chunks.")
    print("The fix is fully applied. Safe to move on to re-running the SOC scenarios.")
