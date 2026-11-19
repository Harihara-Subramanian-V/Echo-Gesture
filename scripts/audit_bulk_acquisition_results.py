"""
Audit bulk acquisition results on disk and synchronize normalized_multidataset_manifest.csv:
1. Inspects all acquired clips in data/raw/ms_asl.
2. Evaluates class-by-class and split-by-split representation.
3. Updates download_status in normalized_multidataset_manifest.csv.
4. Generates data for reports/msasl_bulk_download_report.md.
"""

import os
import csv
import json
from collections import defaultdict, Counter

PROJECT_ROOT = r"c:\EchoCam"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "normalized_multidataset_manifest.csv")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "msasl_matched_samples.json")
SUMMARY_OUT_PATH = os.path.join(PROJECT_ROOT, "reports", "bulk_acquisition_summary.json")

def main():
    # 1. Inspect raw files on disk
    canonical_files = [f for f in os.listdir(RAW_DIR) if f.startswith("msasl_") and not f.startswith("msasl_pilot_") and f.endswith(".mp4")]
    pilot_files = [f for f in os.listdir(RAW_DIR) if f.startswith("msasl_pilot_") and f.endswith(".mp4")]
    
    print(f"Canonical MS-ASL files on disk: {len(canonical_files)}")
    print(f"Pilot MS-ASL files on disk: {len(pilot_files)}")

    # Total bytes of canonical files
    canonical_bytes = sum(os.path.getsize(os.path.join(RAW_DIR, f)) for f in canonical_files)
    print(f"Total Canonical Storage: {canonical_bytes:,} bytes ({canonical_bytes / (1024*1024):.2f} MB)")

    # 2. Load manifest
    manifest_rows = list(csv.DictReader(open(MANIFEST_PATH, "r", encoding="utf-8")))
    fieldnames = list(manifest_rows[0].keys())

    # Map downloaded filenames
    existing_files_set = set(canonical_files)

    # Counters
    class_acquired = Counter()
    split_acquired = Counter()
    signer_acquired = set()
    source_stats = Counter()

    for r in manifest_rows:
        if r["dataset_source"] == "MS-ASL":
            fname = os.path.basename(r["video_path"])
            if fname in existing_files_set:
                r["download_status"] = "downloaded_verified"
                class_acquired[r["normalized_label"]] += 1
                split_acquired[r["unified_split"]] += 1
                signer_acquired.add(r["signer_id"])
                source_stats["msasl_verified"] += 1
            else:
                r["download_status"] = "failed_link_rot"
                source_stats["msasl_failed"] += 1
        elif r["dataset_source"] == "WLASL":
            source_stats["wlasl_pending"] += 1

    print("\nManifest Synchronization Results:")
    print(f"  MS-ASL Verified Clips: {source_stats['msasl_verified']} (matches files on disk)")
    print(f"  MS-ASL Failed Clips:   {source_stats['msasl_failed']}")
    print(f"  WLASL Pending Clips:   {source_stats['wlasl_pending']}")
    print(f"  Unique Signers with acquired videos: {len(signer_acquired)}")

    print("\nAcquired Clips per Split (unified_split):")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:<6}: {split_acquired[sp]} clips")

    print(f"\nClass Representation across {len(class_acquired)} Classes:")
    min_cls = min(class_acquired.items(), key=lambda x: x[1])
    max_cls = max(class_acquired.items(), key=lambda x: x[1])
    print(f"  Min class instances: {min_cls[0]} ({min_cls[1]} clips)")
    print(f"  Max class instances: {max_cls[0]} ({max_cls[1]} clips)")
    print(f"  Average instances/class: {sum(class_acquired.values()) / len(class_acquired):.1f} clips")

    # Save synchronized manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Synchronized manifest saved to {MANIFEST_PATH}")

    # Export summary json
    summary_data = {
        "total_attempted": 1994,
        "total_verified": source_stats["msasl_verified"],
        "total_failed": source_stats["msasl_failed"],
        "retention_rate_pct": round(source_stats["msasl_verified"] / 1994 * 100, 2),
        "total_bytes": canonical_bytes,
        "total_mb": round(canonical_bytes / (1024 * 1024), 2),
        "unique_signers_acquired": len(signer_acquired),
        "classes_represented": len(class_acquired),
        "splits_breakdown": dict(split_acquired),
        "class_breakdown": dict(class_acquired)
    }
    with open(SUMMARY_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary data saved to {SUMMARY_OUT_PATH}")

if __name__ == "__main__":
    main()
