import os
import csv
import json
import random
import cv2

# Define linguistically grounded 1-handed vs 2-handed labels in our 40 classes
ONE_HANDED_SIGNS = {
    "black", "blue", "candy", "cool", "cousin", "deaf", "drink", "fine", 
    "hot", "like", "mother", "no", "orange", "thin", "who", "woman", "yes"
}

TWO_HANDED_SIGNS = {
    "all", "bed", "before", "book", "bowling", "can", "chair", "clothes", 
    "computer", "dog", "family", "finish", "fish", "go", "graduate", 
    "help", "many", "now", "table", "thanksgiving", "walk", "what", "year"
}

def get_video_metadata(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2)
    }

def main():
    manifest_path = "data/splits/normalized_multidataset_manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [r for r in csv.DictReader(f) if r["dataset_source"] == "MS-ASL" and r["download_status"] == "downloaded_verified"]
        
    print(f"Total downloaded verified MS-ASL records: {len(records)}")
    
    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(records)
    
    # We want 20 videos:
    # 10 train (5 one-handed, 5 two-handed)
    # 5 val (2 or 3 one-handed, 3 or 2 two-handed)
    # 5 test (3 or 2 one-handed, 2 or 3 two-handed)
    # Total: 10 one-handed, 10 two-handed
    # All 20 videos must have UNIQUE classes and UNIQUE signers!
    
    selected = []
    selected_classes = set()
    selected_signers = set()
    
    target_quotas = {
        ("train", "1-handed"): 5,
        ("train", "2-handed"): 5,
        ("val", "1-handed"): 2,
        ("val", "2-handed"): 3,
        ("test", "1-handed"): 3,
        ("test", "2-handed"): 2,
    }
    
    current_counts = {k: 0 for k in target_quotas}
    
    for r in records:
        split = r["unified_split"]
        gloss = r["normalized_label"]
        signer = r["signer_id"]
        vpath = os.path.normpath(r["video_path"])
        
        if not os.path.exists(vpath):
            continue
            
        handedness = "1-handed" if gloss in ONE_HANDED_SIGNS else ("2-handed" if gloss in TWO_HANDED_SIGNS else "unknown")
        if handedness == "unknown":
            continue
            
        quota_key = (split, handedness)
        if quota_key not in target_quotas:
            continue
            
        if current_counts[quota_key] >= target_quotas[quota_key]:
            continue
            
        if gloss in selected_classes:
            continue
            
        if signer in selected_signers:
            continue
            
        # Get video dimensions
        vmeta = get_video_metadata(vpath)
        if not vmeta or vmeta["frame_count"] < 10:
            continue
            
        item = {
            "video_id": r["video_id"],
            "dataset": r["dataset_source"],
            "normalized_label": gloss,
            "original_label": r["original_label"],
            "signer_id": signer,
            "unified_split": split,
            "handedness_type": handedness,
            "video_path": vpath,
            **vmeta
        }
        
        selected.append(item)
        selected_classes.add(gloss)
        selected_signers.add(signer)
        current_counts[quota_key] += 1
        
        if len(selected) == 20:
            break
            
    print(f"Selection complete. Selected: {len(selected)} videos.")
    print(f"Unique classes: {len(selected_classes)}, Unique signers: {len(selected_signers)}")
    print(f"Quotas achieved: {current_counts}")
    
    # Sort for clean presentation by split, then gloss
    split_order = {"train": 0, "val": 1, "test": 2}
    selected.sort(key=lambda x: (split_order[x["unified_split"]], x["normalized_label"]))
    
    out_json = "data/metadata/ms_asl/landmark_pilot_candidates.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
        
    print(f"Saved candidate selection to {out_json}")
    
    # Print formatted table
    print("\n| # | Split | Gloss | Handedness | Signer ID | Resolution | FPS | Frames | Duration (s) | Video Path |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for idx, s in enumerate(selected, 1):
        print(f"| {idx} | {s['unified_split']} | {s['normalized_label']} | {s['handedness_type']} | {s['signer_id']} | {s['resolution']} | {s['fps']} | {s['frame_count']} | {s['duration_sec']} | {s['video_path']} |")

if __name__ == "__main__":
    main()
