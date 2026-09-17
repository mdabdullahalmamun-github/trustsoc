import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# MITRE ATT&CK STIX descriptions embed inline citation markers in this
# exact format: "(Citation: Name Here)" -- sometimes with a year, sometimes
# several stacked together. These are real references from MITRE's own
# page (e.g. "(Citation: Harmj0y Roasting AS-REPs Jan 2017)"), not
# fabrications -- but leaving them in the knowledge base text means a
# model reading that chunk as context sees this citation style sitting
# right there in the source material, and naturally echoes it back,
# competing with the [MITRE ATT&CK Txxxx] format the system prompt
# actually requires. Confirmed during Phase 4 SOC-scenario testing: 42%
# of Mistral's and 28% of LLaMA 3's answers contained at least one
# bracketed reference that didn't match the required ID format, and the
# large majority of those strings were real MITRE citation names rather
# than invented ones. Stripped here, at the source, before chunking --
# rather than trying to filter it out of every answer after generation.
_MITRE_CITATION_RE = re.compile(r"\(Citation:\s*[^)]+\)")

def _strip_citation_markers(text):
    """Remove MITRE's inline (Citation: X) markers from a description,
    and tidy up the double-spacing / space-before-punctuation left behind."""
    cleaned = _MITRE_CITATION_RE.sub("", text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned

def load_attack(path=BASE_DIR / "data" / "enterprise-attack.json"):
    """Return a list of {id, text, metadata} dicts from the MITRE ATT&CK STIX bundle."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        # The Txxxx ID lives in external_references where source_name == 'mitre-attack'
        ext = next((e for e in obj.get("external_references", [])
                    if e.get("source_name") == "mitre-attack"), {})
        tech_id = ext.get("external_id", "")
        name = obj.get("name", "")
        desc = _strip_citation_markers(obj.get("description", ""))
        tactics = ", ".join(ph.get("phase_name", "")
                            for ph in obj.get("kill_chain_phases", []))
        text = f"MITRE ATT&CK Technique {tech_id}: {name}. Tactic(s): {tactics}. {desc}"
        chunks.append({
            "id": f"attack-{tech_id}",
            "text": text,
            "metadata": {"source": "MITRE ATT&CK", "technique_id": tech_id,
                         "name": name, "tactics": tactics},
        })
    return chunks

attack_chunks = load_attack()
print(f"Parsed {len(attack_chunks)} ATT&CK techniques")
print("Example:\n", attack_chunks[0]["text"][:300])
