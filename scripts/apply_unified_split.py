"""
Test unified_split generation and verify zero cross-dataset leakage.
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
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    # 1. Map WLASL YouTube IDs to WLASL split
    wlasl_yt_to_split = {}
    for r in rows:
        if r["dataset_source"] == "WLASL":
            yt_id = extract_yt_id(r["source_url"])
            if yt_id:
                wlasl_yt_to_split[yt_id] = r["original_split"]

    # 2. Assign unified_split
    realigned_count = 0
    split_class_counts = defaultdict(lambda: Counter())
    split_instance_counts = Counter()

    for r in rows:
        orig_sp = r["original_split"]
        if r["dataset_source"] == "WLASL":
            r["unified_split"] = orig_sp
        else:
            yt_id = extract_yt_id(r["source_url"])
            if yt_id and yt_id in wlasl_yt_to_split:
                wlasl_sp = wlasl_yt_to_split[yt_id]
                if orig_sp != wlasl_sp:
                    realigned_count += 1
                r["unified_split"] = wlasl_sp
            else:
                r["unified_split"] = orig_sp

        split_instance_counts[r["unified_split"]] += 1
        split_class_counts[r["unified_split"]][r["normalized_label"]] += 1

    print(f"Total rows: {len(rows)}")
    print(f"MS-ASL clips realigned to WLASL split for zero leakage: {realigned_count}")
    print(f"Unified split instance counts: {dict(split_instance_counts)}")
    for sp in ["train", "val", "test"]:
        print(f"  Split '{sp}': {len(split_class_counts[sp])}/40 classes represented, {split_instance_counts[sp]} instances")

    # 3. Verify zero video leakage across unified splits
    yt_split_check = defaultdict(set)
    for r in rows:
        yt_id = extract_yt_id(r["source_url"])
        if yt_id:
            yt_split_check[yt_id].add(r["unified_split"])

    leaks = {k: v for k, v in yt_split_check.items() if len(v) > 1}
    print(f"Video leaks across unified splits: {len(leaks)}")
    assert len(leaks) == 0, "Video leaks exist!"

    # 4. Update the manifest with unified_split column if not present
    new_fieldnames = list(fieldnames)
    if "unified_split" not in new_fieldnames:
        # insert after original_split
        idx = new_fieldnames.index("original_split") + 1
        new_fieldnames.insert(idx, "unified_split")

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully updated normalized manifest with unified_split column!")

if __name__ == "__main__":
    main()
