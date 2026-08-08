import os, subprocess, pathlib

folders = ["data", "kb", "pipeline", "evaluation", "ui", "demo", "benchmarks", "tests", "results", "figures"]
for f in folders:
    pathlib.Path(f).mkdir(exist_ok=True)
print("Folders:", ", ".join(folders))

# A .gitignore so you do not commit gigabytes of model data or raw feeds
gitignore_path = pathlib.Path(".gitignore")
if not gitignore_path.exists():
    gitignore = """trustsoc_env/
__pycache__/
kb/
data/*.json
data/*.csv
*.pyc
.ipynb_checkpoints/
"""
    gitignore_path.write_text(gitignore)
    print("Wrote .gitignore")
else:
    print(".gitignore already exists — left untouched")

# Only write a placeholder README if one doesn't already exist — never overwrite
# a real README that's since been written by hand.
readme_path = pathlib.Path("README.md")
if not readme_path.exists():
    readme = """# TrustSOC
A Multi-Dataset RAG Framework for Faithful Cyber Threat Analysis.
MSc Cybersecurity project — Md Abdullah Al Mamun (2010022), University of Bedfordshire.
Supervisor: Dr Monika Roopak.
"""
    readme_path.write_text(readme)
    print("Wrote placeholder README.md")
else:
    print("README.md already exists — left untouched")

# Initialise git (safe to run even if already initialised)
for cmd in [["git","init"], ["git","add","."],
            ["git","commit","-m","Phase 0: project skeleton"]]:
    try: subprocess.run(cmd, check=False)
    except Exception as e: print("git note:", e)
print("\nNow create an EMPTY public repo named 'trustsoc' on github.com, then run:")
print("  git remote add origin https://github.com/YOUR_USERNAME/trustsoc")
print("  git branch -M main")
print("  git push -u origin main")