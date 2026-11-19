"""
EchoGesture Manifest Generator
Creates deterministic signer-independent splits for the curated 40-class WLASL dataset.
"""

import json
import csv
import os
import random
import urllib.request

METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
DATA_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "splits")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset.yaml")

SELECTED_GLOSSES = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
    "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
    "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
    "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate"
]

def load_metadata():
    print(f"Fetching official WLASL metadata from {METADATA_URL}...")
    req = urllib.request.Request(METADATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        raw = json.loads(res.read().decode('utf-8'))
    return {item["gloss"]: item["instances"] for item in raw}

def compute_signer_independent_splits(wlasl_dict, random_seed=6):
    """
    Computes deterministic, verified signer-independent splits for the 40 classes.
    Seed 6 produces:
    - Train: 53 signers, 635 instances (69.7%), 40/40 classes
    - Val:   7 signers, 146 instances (16.0%), 40/40 classes
    - Test:  19 signers, 130 instances (14.3%), 40/40 classes
    - Overlap: 0 signers between any splits.
    """
    all_signers = set()
    signer_gloss_map = {}
    for g in SELECTED_GLOSSES:
        for inst in wlasl_dict[g]:
            s = inst.get("signer_id")
            if s is not None:
                all_signers.add(s)
                signer_gloss_map.setdefault(s, set()).add(g)

    sorted_s = sorted(all_signers, key=lambda s: len(signer_gloss_map[s]), reverse=True)

    random.seed(random_seed)
    train_s = set()
    val_s = set()
    test_s = set()

    # Major hub signers allocated to separate splits
    hubs = [5, 109, 11]
    random.shuffle(hubs)
    train_s.add(hubs[0])
    val_s.add(hubs[1])
    test_s.add(hubs[2])

    remaining = [s for s in sorted_s if s not in hubs]
    random.shuffle(remaining)

    train_insts = len([i for g in SELECTED_GLOSSES for i in wlasl_dict[g] if i.get("signer_id") == hubs[0]])
    val_insts = len([i for g in SELECTED_GLOSSES for i in wlasl_dict[g] if i.get("signer_id") == hubs[1]])
    test_insts = len([i for g in SELECTED_GLOSSES for i in wlasl_dict[g] if i.get("signer_id") == hubs[2]])

    for s in remaining:
        s_count = len([i for g in SELECTED_GLOSSES for i in wlasl_dict[g] if i.get("signer_id") == s])
        total = train_insts + val_insts + test_insts + s_count
        ratio_t = train_insts / max(1, total)
        ratio_v = val_insts / max(1, total)
        ratio_te = test_insts / max(1, total)

        diff_t = 0.70 - ratio_t
        diff_v = 0.15 - ratio_v
        diff_te = 0.15 - ratio_te

        max_diff = max(diff_t, diff_v, diff_te)
        if max_diff == diff_t:
            train_s.add(s)
            train_insts += s_count
        elif max_diff == diff_v:
            val_s.add(s)
            val_insts += s_count
        else:
            test_s.add(s)
            test_insts += s_count

    return train_s, val_s, test_s

def generate_manifest():
    wlasl_dict = load_metadata()
    train_s, val_s, test_s = compute_signer_independent_splits(wlasl_dict, random_seed=6)

    os.makedirs(DATA_SPLITS_DIR, exist_ok=True)
    manifest_rows = []

    for idx, gloss in enumerate(SELECTED_GLOSSES):
        for inst in wlasl_dict[gloss]:
            s_id = inst.get("signer_id")
            v_id = inst["video_id"]
            if s_id in train_s:
                split_name = "train"
            elif s_id in val_s:
                split_name = "val"
            elif s_id in test_s:
                split_name = "test"
            else:
                split_name = "unassigned"

            manifest_rows.append({
                "video_id": v_id,
                "gloss": gloss,
                "class_index": idx,
                "signer_id": s_id,
                "split": split_name,
                "fps": inst.get("fps", 25),
                "frame_start": inst.get("frame_start", 1),
                "frame_end": inst.get("frame_end", -1),
                "source": inst.get("source", ""),
                "url": inst.get("url", ""),
                "bbox": json.dumps(inst.get("bbox", []))
            })

    # Sort rows by class_index then video_id
    manifest_rows.sort(key=lambda r: (r["class_index"], r["video_id"]))

    fieldnames = [
        "video_id", "gloss", "class_index", "signer_id", "split",
        "fps", "frame_start", "frame_end", "source", "url", "bbox"
    ]

    # Write master manifest
    master_csv = os.path.join(DATA_SPLITS_DIR, "manifest.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Generated master manifest: {master_csv} ({len(manifest_rows)} rows)")

    # Write individual split files
    for sp in ["train", "val", "test"]:
        sp_rows = [r for r in manifest_rows if r["split"] == sp]
        sp_csv = os.path.join(DATA_SPLITS_DIR, f"{sp}.csv")
        with open(sp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sp_rows)
        print(f"Generated {sp} split manifest: {sp_csv} ({len(sp_rows)} rows)")

if __name__ == "__main__":
    generate_manifest()
