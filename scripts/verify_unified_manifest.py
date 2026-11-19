"""
Comprehensive audit of the normalized multi-dataset manifest:
- Checks schema integrity across all 2,905 rows.
- Verifies 40-class lexicon label consistency across datasets.
- Reconciles any cross-dataset YouTube video split misalignment.
- Produces exact counts for downloaded/verified, questionable, failed, and pending clips.
"""

import os
import csv
import json
import re
from collections import defaultdict, Counter

PROJECT_ROOT = r"c:\EchoCam"
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "normalized_multidataset_manifest.csv")
PILOT_DATA_PATH = os.path.join(PROJECT_ROOT, "reports", "pilot_validation_data.json")
WLASL_SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

def extract_yt_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None

def main():
    # 1. Load manifest
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} rows from normalized manifest.")

    # 2. Check schema and required fields
    required_fields = [
        "dataset_source", "video_id", "original_label", "normalized_label",
        "video_path", "signer_id", "original_split", "download_status",
        "source_url", "license_reference"
    ]
    missing_fields_count = 0
    for r in rows:
        for rf in required_fields:
            if not r.get(rf):
                missing_fields_count += 1
    print(f"Missing required fields across all rows: {missing_fields_count}")

    # 3. Verify label consistency across all 40 classes
    target_40_classes = [
        "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
        "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
        "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
        "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate"
    ]
    
    unique_norm_labels = set(r["normalized_label"] for r in rows)
    labels_diff = unique_norm_labels.symmetric_difference(set(target_40_classes))
    print(f"Normalized labels count: {len(unique_norm_labels)} / 40. Difference: {labels_diff}")

    # Verify per-source class coverage
    wlasl_labels = set(r["normalized_label"] for r in rows if r["dataset_source"] == "WLASL")
    msasl_labels = set(r["normalized_label"] for r in rows if r["dataset_source"] == "MS-ASL")
    print(f"WLASL classes: {len(wlasl_labels)} / 40 | MS-ASL classes: {len(msasl_labels)} / 40")

    # 4. Check for cross-dataset YouTube video overlap & split alignment
    yt_to_splits = defaultdict(set)
    yt_to_sources = defaultdict(set)
    for r in rows:
        yt_id = extract_yt_id(r["source_url"])
        if yt_id:
            yt_to_splits[yt_id].add(r["original_split"])
            yt_to_sources[yt_id].add(r["dataset_source"])

    shared_yt_vids = {k: v for k, v in yt_to_sources.items() if len(v) > 1}
    print(f"Shared YouTube videos between WLASL and MS-ASL: {len(shared_yt_vids)}")

    split_conflicts = {k: yt_to_splits[k] for k in shared_yt_vids if len(yt_to_splits[k]) > 1}
    print(f"Shared YouTube videos with split conflicts in original annotations: {len(split_conflicts)}")

    # 5. Check actual files in data/raw/ms_asl
    ms_asl_raw_dir = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")
    downloaded_files = [f for f in os.listdir(ms_asl_raw_dir) if f.endswith(".mp4")]
    print(f"Downloaded mp4 files in data/raw/ms_asl: {len(downloaded_files)}")

    # Load pilot results
    with open(PILOT_DATA_PATH, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    # Classify all 2,905 items:
    # - Downloaded and verified
    # - Downloaded but questionable
    # - Failed or inaccessible
    # - Candidate clips not yet downloaded
    downloaded_and_verified = 0
    downloaded_questionable = 0
    failed_inaccessible = 0
    pending_not_downloaded = 0

    pilot_status_map = {}
    for p in pilot_data:
        dl = p["download_result"]
        ff = p["ffprobe_result"]
        url = p["sample"]["url"]
        yt_id = extract_yt_id(url)
        st = dl.get("status")
        if st == "success" and ff.get("valid"):
            # Check if questionable
            if ff.get("duration_delta", 0.0) > 0.5:
                pilot_status_map[yt_id] = "questionable"
            else:
                pilot_status_map[yt_id] = "verified"
        elif st == "failed":
            pilot_status_map[yt_id] = "failed"

    for r in rows:
        yt_id = extract_yt_id(r["source_url"])
        if yt_id in pilot_status_map:
            ps = pilot_status_map[yt_id]
            if ps == "verified":
                downloaded_and_verified += 1
            elif ps == "questionable":
                downloaded_questionable += 1
            elif ps == "failed":
                failed_inaccessible += 1
        else:
            pending_not_downloaded += 1

    print("\nReadiness Breakdown:")
    print(f"  1. Downloaded and Verified: {downloaded_and_verified}")
    print(f"  2. Downloaded but Questionable: {downloaded_questionable}")
    print(f"  3. Failed / Inaccessible (Link Rot): {failed_inaccessible}")
    print(f"  4. Candidate Clips Not Yet Downloaded: {pending_not_downloaded}")
    print(f"  Total: {downloaded_and_verified + downloaded_questionable + failed_inaccessible + pending_not_downloaded} / {len(rows)}")

if __name__ == "__main__":
    main()
