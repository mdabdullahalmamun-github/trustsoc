import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_nvd(
    paths=(
        BASE_DIR / "data" / "nvd_2023.json",
        BASE_DIR / "data" / "nvd_2024.json",
    ),
    severities=("CRITICAL", "HIGH"),
):
    """Parse NVD API 2.0 JSON feeds, keeping only CRITICAL/HIGH CVEs."""
    chunks = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  (skip, not found: {path})"); continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            descs = cve.get("descriptions", [])
            desc = next((d["value"] for d in descs if d.get("lang") == "en"), "")
            # severity from CVSS v3.1 if present
            sev = ""
            metrics = cve.get("metrics", {}).get("cvssMetricV31", [])
            if metrics:
                sev = metrics[0].get("cvssData", {}).get("baseSeverity", "")
            if sev and sev not in severities:
                continue
            chunks.append({
                "id": f"cve-{cve_id}",
                "text": f"Vulnerability {cve_id} (severity {sev or 'n/a'}): {desc}",
                "metadata": {"source": "NVD", "cve_id": cve_id, "severity": sev},
            })
    return chunks

def load_kev(path=BASE_DIR / "data" / "cisa_kev.csv"):
    """Parse the CISA Known Exploited Vulnerabilities CSV."""
    chunks = []
    p = Path(path)
    if not p.exists():
        print(f"  (skip, not found: {path})"); return chunks
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("cveID", "")
            chunks.append({
                "id": f"kev-{cid}",
                "text": (f"Actively exploited vulnerability {cid}: "
                         f"{row.get('vulnerabilityName','')}. "
                         f"{row.get('shortDescription','')} "
                         f"Product: {row.get('vendorProject','')} {row.get('product','')}."),
                "metadata": {"source": "CISA KEV", "cve_id": cid,
                             "date_added": row.get("dateAdded", "")},
            })
    return chunks

nvd_chunks = load_nvd()
kev_chunks = load_kev()
print(f"NVD CVEs: {len(nvd_chunks)}   CISA KEV: {len(kev_chunks)}")