import os
import csv
import glob

def main():
    print("=== STEP 1: DATASET VERIFICATION ===")
    
    # 1. Check data/raw/wlasl
    wlasl_dir = os.path.join("data", "raw", "wlasl")
    wlasl_files = glob.glob(os.path.join(wlasl_dir, "*.mp4")) if os.path.exists(wlasl_dir) else []
    print(f"Physical WLASL videos in {wlasl_dir}: {len(wlasl_files)}")
    
    # 2. Check data/raw/ms_asl
    ms_asl_dir = os.path.join("data", "raw", "ms_asl")
    ms_asl_files = [f for f in os.listdir(ms_asl_dir) if f.endswith(".mp4")] if os.path.exists(ms_asl_dir) else []
    print(f"Physical MS-ASL videos in {ms_asl_dir}: {len(ms_asl_files)} (1452 bulk + 14 pilot)")
    
    # 3. Read normalized_multidataset_manifest.csv
    manifest_path = os.path.join("data", "splits", "normalized_multidataset_manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found!")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    print(f"Total records in normalized manifest: {len(records)}")
    
    status_counts = {}
    dataset_counts = {}
    split_counts = {}
    missing_physical = []
    found_physical = []
    
    for r in records:
        ds = r.get("dataset_source", "")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        status = r.get("download_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
        
        if ds == "MS-ASL" and status == "downloaded_verified":
            vpath = r.get("video_path", "")
            norm_vpath = os.path.normpath(vpath)
            if os.path.exists(norm_vpath):
                found_physical.append((norm_vpath, r))
                split = r.get("unified_split", "")
                split_counts[split] = split_counts.get(split, 0) + 1
            else:
                missing_physical.append(vpath)
                
    print(f"Dataset breakdown: {dataset_counts}")
    print(f"Status breakdown: {status_counts}")
    print(f"Physical MS-ASL verified files found on disk: {len(found_physical)}")
    print(f"Missing physical MS-ASL verified files: {len(missing_physical)}")
    print(f"Unified split breakdown for verified MS-ASL: {split_counts}")
    
    # 4. Check original WLASL splits preservation
    wlasl_manifest = os.path.join("data", "splits", "manifest.csv")
    wlasl_train = os.path.join("data", "splits", "train.csv")
    wlasl_val = os.path.join("data", "splits", "val.csv")
    wlasl_test = os.path.join("data", "splits", "test.csv")
    
    for p in [wlasl_manifest, wlasl_train, wlasl_val, wlasl_test]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            print(f"Preserved {p}: {row_count} rows")
        else:
            print(f"WARNING: {p} missing!")

if __name__ == "__main__":
    main()
