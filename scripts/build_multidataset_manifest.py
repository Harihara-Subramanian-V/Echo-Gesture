"""
EchoGesture Multi-Dataset Manifest and Overlap Processor
Strictly operates using Python standard libraries (json, csv, os, re).
"""

import os
import json
import csv
from collections import defaultdict, Counter

PROJECT_ROOT = r"c:\EchoCam"
WLASL_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "manifest.csv")
MSASL_MATCHED_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "msasl_matched_samples.json")
OUTPUT_OVERLAP_CSV = os.path.join(PROJECT_ROOT, "reports", "vocabulary_overlap.csv")
OUTPUT_NORMALIZED_MANIFEST = os.path.join(PROJECT_ROOT, "data", "splits", "normalized_multidataset_manifest.csv")

def main():
    # 1. Load WLASL Manifest
    wlasl_samples = []
    with open(WLASL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            wlasl_samples.append(r)

    print(f"Loaded {len(wlasl_samples)} WLASL samples from {WLASL_MANIFEST_PATH}")

    # WLASL breakdown by class
    wlasl_class_counts = Counter()
    wlasl_class_signers = defaultdict(set)
    for r in wlasl_samples:
        gloss = r["gloss"]
        wlasl_class_counts[gloss] += 1
        wlasl_class_signers[gloss].add(r["signer_id"])

    # 2. Load MS-ASL Matched Samples
    with open(MSASL_MATCHED_PATH, "r", encoding="utf-8") as f:
        msasl_samples = json.load(f)

    print(f"Loaded {len(msasl_samples)} MS-ASL matched samples from {MSASL_MATCHED_PATH}")

    # MS-ASL breakdown by class
    msasl_class_counts = Counter()
    msasl_class_signers = defaultdict(set)
    msasl_splits_count = Counter()
    for s in msasl_samples:
        gloss = s["target_gloss"]
        msasl_class_counts[gloss] += 1
        msasl_class_signers[gloss].add(s["signer_id"])
        msasl_splits_count[s.get("split", "unknown")] += 1

    print(f"MS-ASL splits breakdown: {dict(msasl_splits_count)}")

    # 3. Build vocabulary_overlap.csv
    # 40 target classes ordered as in configs/dataset.yaml (or sorted)
    classes_40 = [
        "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
        "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
        "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
        "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate"
    ]

    fieldnames = [
        "class_index",
        "gloss",
        "wlasl_instances",
        "wlasl_signers",
        "msasl_instances",
        "msasl_signers",
        "asl_citizen_overlap",
        "how2sign_overlap",
        "total_instances",
        "total_unique_signers_est"
    ]

    rows = []
    total_wlasl_inst = 0
    total_msasl_inst = 0

    for idx, gloss in enumerate(classes_40):
        w_cnt = wlasl_class_counts.get(gloss, 0)
        w_sig = len(wlasl_class_signers.get(gloss, set()))
        m_cnt = msasl_class_counts.get(gloss, 0)
        m_sig = len(msasl_class_signers.get(gloss, set()))
        
        total_wlasl_inst += w_cnt
        total_msasl_inst += m_cnt

        rows.append({
            "class_index": idx,
            "gloss": gloss,
            "wlasl_instances": w_cnt,
            "wlasl_signers": w_sig,
            "msasl_instances": m_cnt,
            "msasl_signers": m_sig,
            "asl_citizen_overlap": "Present in 2,731 Lexicon (Selective Blocked)",
            "how2sign_overlap": "Present in Corpus (Continuous Deferred)",
            "total_instances": w_cnt + m_cnt,
            "total_unique_signers_est": w_sig + m_sig  # Disjoint signer pools between WLASL & MS-ASL
        })

    with open(OUTPUT_OVERLAP_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote vocabulary overlap CSV to {OUTPUT_OVERLAP_CSV}")
    print(f"Summary: 40 classes, WLASL total={total_wlasl_inst}, MS-ASL total={total_msasl_inst}, Combined={total_wlasl_inst + total_msasl_inst}")

    # 4. Generate data/splits/normalized_multidataset_manifest.csv
    # Fields:
    # dataset_source,video_id,original_label,normalized_label,video_path,signer_id,original_split,download_status,source_url,license_reference
    
    norm_fieldnames = [
        "dataset_source",
        "video_id",
        "original_label",
        "normalized_label",
        "video_path",
        "signer_id",
        "original_split",
        "download_status",
        "source_url",
        "license_reference"
    ]

    norm_rows = []

    # A. Add WLASL entries
    for r in wlasl_samples:
        vid = r["video_id"]
        gloss = r["gloss"]
        split = r["split"]
        signer = f"wlasl_{r['signer_id']}"
        source_url = r.get("url", "")
        # video_path where it will reside when imported
        vid_rel_path = f"data/raw/wlasl/{vid}.mp4"
        norm_rows.append({
            "dataset_source": "WLASL",
            "video_id": f"wlasl_{vid}",
            "original_label": gloss,
            "normalized_label": gloss,
            "video_path": vid_rel_path,
            "signer_id": signer,
            "original_split": split,
            "download_status": "pending_request_missing_txt",
            "source_url": source_url,
            "license_reference": "CC BY-NC 4.0 / C-UDA 1.0"
        })

    # B. Add MS-ASL entries
    # MS-ASL unique ID generation: msasl_{split}_{index}_{label}
    for idx, s in enumerate(msasl_samples):
        gloss = s["target_gloss"]
        split = s.get("split", "train")
        signer = f"msasl_{s['signer_id']}"
        url = s.get("url", "")
        clean_text = s.get("clean_text", gloss)
        start_t = s.get("start_time", 0.0)
        end_t = s.get("end_time", 0.0)
        
        # generate a unique, clean video_id
        # YouTube video ID extracted from URL if possible
        yt_id = url.split("v=")[-1].split("&")[0] if "v=" in url else f"clip_{idx:05d}"
        unique_vid = f"msasl_{split}_{idx:04d}_{yt_id}"
        vid_rel_path = f"data/raw/ms_asl/{unique_vid}.mp4"

        norm_rows.append({
            "dataset_source": "MS-ASL",
            "video_id": unique_vid,
            "original_label": clean_text,
            "normalized_label": gloss,
            "video_path": vid_rel_path,
            "signer_id": signer,
            "original_split": split,
            "download_status": "metadata_verified_pending_download",
            "source_url": f"{url}#t={start_t},{end_t}",
            "license_reference": "Computational Use of Data Agreement (C-UDA) 0.1"
        })

    with open(OUTPUT_NORMALIZED_MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=norm_fieldnames)
        writer.writeheader()
        writer.writerows(norm_rows)

    print(f"Wrote normalized multi-dataset manifest to {OUTPUT_NORMALIZED_MANIFEST}")
    print(f"Total entries in normalized manifest: {len(norm_rows)} ({len(wlasl_samples)} WLASL + {len(msasl_samples)} MS-ASL)")

if __name__ == "__main__":
    main()
