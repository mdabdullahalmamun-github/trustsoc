"""
An interactive, browsable explorer for the TrustSOC knowledge base --
complements the embedding scatter plot by showing actual CONTENT rather
than abstract geometry: real record counts, real breakdowns by category,
and a searchable table of the real chunks themselves.

This is a standalone Streamlit page, separate from the main SOC dashboard,
so it won't interfere with anything already built. Good for demonstrations: you
can search live, e.g. type "PowerShell" or "ransomware" and show the
actual matching records on screen.

Requirements: streamlit, pandas (both already in the project).

Run from project root:
    streamlit run ui/kb_explorer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TrustSOC Knowledge Base Explorer", layout="wide")


@st.cache_data(show_spinner="Loading knowledge base (only happens once per session)...")
def load_kb():
    client = chromadb.PersistentClient(path="kb")
    collection = client.get_collection(name="cti")
    total = collection.count()

    # Same batching fix as the embedding-plot script -- avoids the
    # "too many SQL variables" error on the full 35,111-row collection.
    BATCH_SIZE = 2000
    ids, metas, docs = [], [], []
    offset = 0
    while offset < total:
        batch = collection.get(include=["metadatas", "documents"], limit=BATCH_SIZE, offset=offset)
        ids.extend(batch["ids"])
        metas.extend(batch["metadatas"])
        docs.extend(batch["documents"])
        offset += BATCH_SIZE

    rows = []
    for _id, meta, doc in zip(ids, metas, docs):
        rows.append({
            "id": _id,
            "source": meta.get("source", "unknown"),
            "technique_id": meta.get("technique_id", ""),
            "name": meta.get("name", ""),
            "tactics": meta.get("tactics", ""),
            "cve_id": meta.get("cve_id", ""),
            "severity": meta.get("severity", ""),
            "text": doc,
        })
    return pd.DataFrame(rows)


def main():
    st.title("TrustSOC Knowledge Base Explorer")
    st.caption("Real content from the actual 35,111-chunk ChromaDB collection -- not a sample.")

    df = load_kb()

    # --- Top-level stats ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total chunks", f"{len(df):,}")
    col2.metric("MITRE ATT&CK techniques", f"{(df['source'] == 'MITRE ATT&CK').sum():,}")
    col3.metric("NVD CVEs", f"{(df['source'] == 'NVD').sum():,}")
    col4.metric("CISA KEV entries", f"{(df['source'] == 'CISA KEV').sum():,}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["ATT&CK Tactic Breakdown", "NVD/KEV Severity Breakdown", "Search the Knowledge Base"])

    with tab1:
        attack_df = df[df["source"] == "MITRE ATT&CK"].copy()
        # tactics field is a comma-separated string per chunk; explode it
        # out so each tactic is counted properly even where a technique
        # belongs to more than one tactic.
        tactic_counts = (
            attack_df["tactics"].str.split(",").explode().str.strip().value_counts()
        )
        st.bar_chart(tactic_counts)
        st.caption(f"{len(attack_df)} techniques across {len(tactic_counts)} tactics.")

    with tab2:
        vuln_df = df[df["source"].isin(["NVD", "CISA KEV"])].copy()
        sev_counts = vuln_df.groupby(["source", "severity"]).size().unstack(fill_value=0)
        st.bar_chart(sev_counts)
        st.caption("Severity breakdown across NVD and CISA KEV entries.")

    with tab3:
        query = st.text_input("Search (matches against chunk text, technique/CVE ID, or name):", "")
        source_filter = st.multiselect("Filter by source:", options=df["source"].unique().tolist(),
                                         default=df["source"].unique().tolist())

        filtered = df[df["source"].isin(source_filter)]
        if query:
            mask = (
                filtered["text"].str.contains(query, case=False, na=False)
                | filtered["technique_id"].str.contains(query, case=False, na=False)
                | filtered["cve_id"].str.contains(query, case=False, na=False)
                | filtered["name"].str.contains(query, case=False, na=False)
            )
            filtered = filtered[mask]

        st.write(f"{len(filtered):,} matching chunks")
        st.dataframe(
            filtered[["source", "technique_id", "cve_id", "name", "severity", "text"]].head(500),
            use_container_width=True,
            height=500,
        )
        if len(filtered) > 500:
            st.caption("Showing first 500 matches -- narrow your search to see more precisely.")


if __name__ == "__main__":
    main()
