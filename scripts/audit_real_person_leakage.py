"""
Independent audit of real-person identity leakage and video overlap between WLASL and MS-ASL.
Audits:
1. YouTube video ID leakage across unified_split.
2. Channel and multi-clip signer clustering (e.g. Dr. Bill Vicars = wlasl_5 & msasl_0).
3. Verifies that WLASL's original splits are completely preserved.
"""

import os
import csv
import re
from collections import defaultdict, Counter

PROJECT_ROOT = r"c:\EchoCam"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "normalized_multidataset_manifest.csv")
WLASL_ORIG_MANIFEST = os.path.join(PROJECT_ROOT, "data", "splits", "manifest.csv")

def extract_yt_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None

def main():
    rows = list(csv.DictReader(open(MANIFEST_PATH, "r", encoding="utf-8")))
    wlasl_orig = list(csv.DictReader(open(WLASL_ORIG_MANIFEST, "r", encoding="utf-8")))

    print(f"Total rows in normalized manifest: {len(rows)}")
    print(f"Total rows in original WLASL manifest: {len(wlasl_orig)}")

    # 1. Verify WLASL original splits are 100% preserved
    wlasl_orig_splits = {r["video_id"]: r["split"] for r in wlasl_orig}
    wlasl_norm_splits = {r["video_id"].replace("wlasl_", ""): r["original_split"] for r in rows if r["dataset_source"] == "WLASL"}
    wlasl_unified_splits = {r["video_id"].replace("wlasl_", ""): r["unified_split"] for r in rows if r["dataset_source"] == "WLASL"}

    mismatch_orig = sum(1 for vid, sp in wlasl_orig_splits.items() if wlasl_norm_splits.get(vid) != sp)
    mismatch_unified = sum(1 for vid, sp in wlasl_orig_splits.items() if wlasl_unified_splits.get(vid) != sp)

    print(f"WLASL original_split mismatch against baseline: {mismatch_orig} (target: 0)")
    # Note on unified: Eya_OdUAevw was unified to 'train' to prevent test/val leakage
    print(f"WLASL unified_split variations from baseline: {mismatch_unified}")

    # 2. Check identical YouTube video ID leakage across unified_split
    yt_unified_splits = defaultdict(set)
    yt_orig_splits = defaultdict(set)
    for r in rows:
        y = extract_yt_id(r["source_url"])
        if y:
            yt_unified_splits[y].add(r["unified_split"])
            yt_orig_splits[y].add(r["original_split"])

    orig_video_leaks = {y: s for y, s in yt_orig_splits.items() if len(s) > 1}
    unified_video_leaks = {y: s for y, s in yt_unified_splits.items() if len(s) > 1}

    print(f"Original multi-dataset video leaks (if merged naively): {len(orig_video_leaks)} videos!")
    print(f"Unified split video leaks (after reconciliation): {len(unified_video_leaks)} videos (target: 0)")
    assert len(unified_video_leaks) == 0, "Video leaks exist in unified_split!"

    # 3. Known Real-Person Signer Cluster Audit: Dr. Bill Vicars
    # In WLASL: signer_id = 'wlasl_5'
    # In MS-ASL: signer_id = 'msasl_0'
    wlasl_5_splits = set(r["unified_split"] for r in rows if r["signer_id"] == "wlasl_5")
    msasl_0_splits = set(r["unified_split"] for r in rows if r["signer_id"] == "msasl_0")

    print(f"\nReal-Person Signer Cluster: Dr. Bill Vicars")
    print(f"  wlasl_5 unified splits: {wlasl_5_splits}")
    print(f"  msasl_0 unified splits: {msasl_0_splits}")
    print(f"  wlasl_5 instance count: {sum(1 for r in rows if r['signer_id'] == 'wlasl_5')}")
    print(f"  msasl_0 instance count: {sum(1 for r in rows if r['signer_id'] == 'msasl_0')}")

    # Notice whether msasl_0 is present in train, val, or test
    msasl_0_orig_splits = set(r["original_split"] for r in rows if r["signer_id"] == "msasl_0")
    print(f"  msasl_0 original splits in MS-ASL: {msasl_0_orig_splits}")

    # 4. Signer ID namespace audit
    # Ensure all signer IDs have explicit dataset prefixes
    invalid_prefixes = [r["signer_id"] for r in rows if not (r["signer_id"].startswith("wlasl_") or r["signer_id"].startswith("msasl_"))]
    print(f"\nSigner IDs without dataset namespace prefix: {len(invalid_prefixes)} (target: 0)")

    # 5. Signer distribution across unified splits
    train_signers = set(r["signer_id"] for r in rows if r["unified_split"] == "train")
    val_signers = set(r["signer_id"] for r in rows if r["unified_split"] == "val")
    test_signers = set(r["signer_id"] for r in rows if r["unified_split"] == "test")

    print(f"\nSigner Count by Unified Split (Namespaced IDs):")
    print(f"  Train signers: {len(train_signers)}")
    print(f"  Val signers:   {len(val_signers)}")
    print(f"  Test signers:  {len(test_signers)}")
    print(f"  Overlap Train-Val:  {len(train_signers.intersection(val_signers))}")
    print(f"  Overlap Train-Test: {len(train_signers.intersection(test_signers))}")
    print(f"  Overlap Val-Test:   {len(val_signers.intersection(test_signers))}")

if __name__ == "__main__":
    main()
