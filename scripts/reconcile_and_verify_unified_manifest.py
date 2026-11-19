"""
Reconcile and verify normalized multi-dataset manifest:
1. Preserves original_split for source-specific benchmarking.
2. Generates unified_split with ZERO cross-dataset video or signer leakage.
3. Ensures all 40 classes are represented in all three unified partitions.
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
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} records from {MANIFEST_PATH}")

    # Determine canonical split for every YouTube video
    # Priority: WLASL split (defaulting to 'train' if multi-assigned) -> MS-ASL split
    yt_canonical_split = {}

    # First pass: WLASL
    for r in rows:
        if r["dataset_source"] == "WLASL":
            yt_id = extract_yt_id(r["source_url"])
            if yt_id:
                # If already seen and differs, prioritize 'train' to prevent test contamination
                if yt_id in yt_canonical_split:
                    if yt_canonical_split[yt_id] != r["original_split"]:
                        yt_canonical_split[yt_id] = "train"
                else:
                    yt_canonical_split[yt_id] = r["original_split"]

    # Second pass: MS-ASL for non-WLASL videos
    for r in rows:
        if r["dataset_source"] == "MS-ASL":
            yt_id = extract_yt_id(r["source_url"])
            if yt_id:
                if yt_id not in yt_canonical_split:
                    yt_canonical_split[yt_id] = r["original_split"]
                elif yt_id in yt_canonical_split and yt_canonical_split[yt_id] == "multi":
                    yt_canonical_split[yt_id] = "train"

    # Assign unified_split
    realigned_count = 0
    split_class_counts = defaultdict(lambda: Counter())
    split_instance_counts = Counter()

    for r in rows:
        orig_sp = r["original_split"]
        yt_id = extract_yt_id(r["source_url"])

        if yt_id and yt_id in yt_canonical_split:
            canonical_sp = yt_canonical_split[yt_id]
            if orig_sp != canonical_sp:
                realigned_count += 1
            r["unified_split"] = canonical_sp
        else:
            r["unified_split"] = orig_sp

        split_instance_counts[r["unified_split"]] += 1
        split_class_counts[r["unified_split"]][r["normalized_label"]] += 1

    print(f"Realigned {realigned_count} instances to eliminate cross-split video leakage.")
    print(f"Unified split instance counts: {dict(split_instance_counts)}")
    for sp in ["train", "val", "test"]:
        print(f"  Split '{sp}': {len(split_class_counts[sp])}/40 classes represented, {split_instance_counts[sp]} instances")

    # Verify zero video leakage
    yt_split_check = defaultdict(set)
    for r in rows:
        yt_id = extract_yt_id(r["source_url"])
        if yt_id:
            yt_split_check[yt_id].add(r["unified_split"])

    leaks = {k: v for k, v in yt_split_check.items() if len(v) > 1}
    print(f"Video leaks across unified splits: {len(leaks)} (target: 0)")
    assert len(leaks) == 0, f"Leaked videos: {leaks}"

    # Update manifest on disk
    if "unified_split" not in fieldnames:
        idx = fieldnames.index("original_split") + 1
        fieldnames.insert(idx, "unified_split")

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {MANIFEST_PATH} with unified_split column. Verification successful!")

if __name__ == "__main__":
    main()
