import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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
        desc = obj.get("description", "")
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