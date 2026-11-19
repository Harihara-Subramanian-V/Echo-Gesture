"""
Graph-based Connected Components Analysis of Real-Person Signer Clusters across WLASL and MS-ASL.
Audits cross-dataset signer identity linkages, shared video nodes, and split assignments.
"""

import os
import csv
import re
from collections import defaultdict, Counter

PROJECT_ROOT = r"c:\EchoCam"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "normalized_multidataset_manifest.csv")

def extract_yt_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None

def main():
    rows = list(csv.DictReader(open(MANIFEST_PATH, "r", encoding="utf-8")))

    # Build bipartite graph between Signers and Videos
    adj = defaultdict(set)
    signer_nodes = set()
    video_nodes = set()
    signer_dataset = {}

    for r in rows:
        sig = r["signer_id"]
        signer_nodes.add(sig)
        signer_dataset[sig] = r["dataset_source"]

        yt_id = extract_yt_id(r["source_url"])
        if yt_id:
            v_node = f"yt_{yt_id}"
            video_nodes.add(v_node)
            adj[sig].add(v_node)
            adj[v_node].add(sig)

    print(f"Total Signer Nodes: {len(signer_nodes)}")
    print(f"Total Video Nodes: {len(video_nodes)}")

    # Find connected components (real-person clusters)
    visited = set()
    clusters = []

    for s in signer_nodes:
        if s not in visited:
            component = []
            queue = [s]
            visited.add(s)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(component)

    print(f"Total Connected Component Clusters: {len(clusters)}")

    # Analyze cross-dataset clusters (clusters with signers from BOTH WLASL and MS-ASL)
    cross_clusters = []
    for c in clusters:
        c_signers = [n for n in c if not n.startswith("yt_")]
        c_videos = [n for n in c if n.startswith("yt_")]
        sources = set(signer_dataset[s] for s in c_signers)
        if len(sources) > 1:
            cross_clusters.append({
                "signers": c_signers,
                "videos": c_videos,
                "sources": sources
            })

    print(f"\nDiscovered {len(cross_clusters)} Cross-Dataset Real-Person Signer Clusters:")
    for idx, cc in enumerate(cross_clusters):
        w_sigs = [s for s in cc["signers"] if s.startswith("wlasl_")]
        m_sigs = [s for s in cc["signers"] if s.startswith("msasl_")]
        print(f"  Cluster {idx+1}: {len(cc['signers'])} signers ({w_sigs} <-> {m_sigs}), {len(cc['videos'])} shared videos")

    # Audit split distribution of each cross-dataset cluster in WLASL baseline vs MS-ASL baseline
    print("\nSplit Allocation Conflict Analysis for Cross-Dataset Clusters:")
    for idx, cc in enumerate(cross_clusters):
        c_signers = set(cc["signers"])
        w_splits = set(r["original_split"] for r in rows if r["signer_id"] in c_signers and r["dataset_source"] == "WLASL")
        m_splits = set(r["original_split"] for r in rows if r["signer_id"] in c_signers and r["dataset_source"] == "MS-ASL")
        w_sigs = [s for s in cc["signers"] if s.startswith("wlasl_")]
        m_sigs = [s for s in cc["signers"] if s.startswith("msasl_")]
        has_conflict = len(w_splits.intersection(m_splits)) == 0 if (w_splits and m_splits) else False
        status = "SPLIT CONFLICT" if has_conflict else "ALIGNED"
        print(f"  Cluster {idx+1} ({w_sigs} <-> {m_sigs}):")
        print(f"    WLASL original split: {w_splits} | MS-ASL original split: {m_splits} -> [{status}]")

if __name__ == "__main__":
    main()
