"""
Interactive, full-database visualisation of the TrustSOC knowledge base,
built specifically for live demonstration  -- not a sample,
the real, complete 35,111 chunks.

Produces a single, self-contained HTML file with no internet dependency
(Plotly.js is embedded directly in the file), so it opens instantly and
reliably in any browser even with no wifi in the room. You can zoom, pan,
and hover over any point to see exactly which technique/CVE it is.

Requirements (install once):
    pip install umap-learn plotly --break-system-packages

Why UMAP instead of t-SNE here: t-SNE gets very slow well before 35,111
points (could take 20-30+ minutes on a laptop). UMAP scales far better to
this size and still produces well-separated clusters, so it's the right
tool specifically for visualising the FULL database rather than a sample.

Run from your project root:
    python evaluation/visualise_kb_full.py

Output:
    evaluation/kb_full_interactive.html   <- open this in any browser
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import numpy as np
import umap
import plotly.graph_objects as go

RANDOM_SEED = 42

COLORS = {
    "MITRE ATT&CK": "#3E5C9A",
    "NVD": "#C6862B",
    "CISA KEV": "#B23A3F",
}


def hover_label(meta, doc_text):
    source = meta.get("source", "unknown")
    if source == "MITRE ATT&CK":
        ident = meta.get("technique_id", "?")
        name = meta.get("name", "")
        return f"<b>{ident}</b> - {name}<br>{source}"
    else:
        ident = meta.get("cve_id", "?")
        sev = meta.get("severity", "n/a")
        snippet = (doc_text[:90] + "...") if doc_text and len(doc_text) > 90 else (doc_text or "")
        return f"<b>{ident}</b> (severity: {sev})<br>{source}<br>{snippet}"


def main():
    client = chromadb.PersistentClient(path="kb")
    collection = client.get_collection(name="cti")
    total = collection.count()
    print(f"Collection has {total} chunks total -- loading all of them.")
    print("This will take a little while for the full database; that's expected.")

    # ChromaDB's underlying SQLite store hits a "too many SQL variables"
    # error if you fetch tens of thousands of rows in one unbatched call
    # (the exact same limit documented in build_kb.py's idempotent-indexing
    # fix, Chapter 4.7). Fetch in batches instead, same pattern as there.
    BATCH_SIZE = 2000
    ids, embeddings_list, metas, docs = [], [], [], []
    offset = 0
    while offset < total:
        batch = collection.get(
            include=["embeddings", "metadatas", "documents"],
            limit=BATCH_SIZE,
            offset=offset,
        )
        ids.extend(batch["ids"])
        embeddings_list.extend(batch["embeddings"])
        metas.extend(batch["metadatas"])
        docs.extend(batch["documents"])
        offset += BATCH_SIZE
        print(f"  Loaded {min(offset, total)}/{total} chunks...")

    embeddings = np.array(embeddings_list)

    print(f"Loaded {len(ids)} chunks, embedding shape {embeddings.shape}.")
    print("Running UMAP on the full set (faster than t-SNE at this scale, "
          "but still expect a few minutes)...")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(embeddings)

    print("Building the interactive plot...")
    sources = [m.get("source", "unknown") for m in metas]
    hover_texts = [hover_label(m, d) for m, d in zip(metas, docs)]

    fig = go.Figure()
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        fig.add_trace(go.Scattergl(
            x=X_2d[idx, 0],
            y=X_2d[idx, 1],
            mode="markers",
            name=f"{src} (n={len(idx)})",
            marker=dict(size=4, color=COLORS.get(src, "#888888"), opacity=0.6),
            text=[hover_texts[i] for i in idx],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"TrustSOC Knowledge Base -- full {total}-chunk embedding space (UMAP projection)",
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        width=1100,
        height=800,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_full_interactive.html")
    # include_plotlyjs='inline' embeds the plotting library directly in the
    # file, so it works with zero internet connection in the room.
    fig.write_html(out_path, include_plotlyjs="inline")

    print(f"\nSaved interactive visualisation to {out_path}")
    print("Double-click that file to open it in your browser -- test it once")
    print("beforehand (ideally with wifi off) to confirm it opens instantly")
    print("with no internet needed, before relying on it in the room.")


if __name__ == "__main__":
    main()
