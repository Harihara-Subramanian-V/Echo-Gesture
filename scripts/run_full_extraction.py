"""
EchoGesture: Full-Scale MediaPipe Landmark Extraction (v3 Audited)
Processes all 1,452 verified MS-ASL videos with batch checkpointing,
resume capability, numerical verification, and failure logging.
"""

import os
import sys
import time
import json
import csv
import shutil
from typing import Dict, List, Any

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.extract_landmarks import LandmarkExtractor


def is_valid_npz(npz_path: str) -> bool:
    """Check if an existing .npz file is complete, valid, and finite."""
    if not os.path.exists(npz_path) or os.path.getsize(npz_path) == 0:
        return False
    try:
        with np.load(npz_path) as data:
            if "flat_features" not in data or "norm_landmarks" not in data:
                return False
            flat = data["flat_features"]
            if len(flat.shape) != 2 or flat.shape[1] != 134:
                return False
            if np.isnan(flat).any() or np.isinf(flat).any():
                return False
        return True
    except Exception:
        return False


def main():
    batch_size = 200
    manifest_path = "data/splits/normalized_multidataset_manifest.csv"
    output_dir = "data/processed/landmarks_v3"
    checkpoint_log_path = "reports/full_extraction_checkpoints.log"
    checkpoint_json_path = os.path.join(output_dir, "checkpoints.json")
    feature_manifest_path = os.path.join(output_dir, "feature_manifest.csv")
    failures_csv_path = "reports/full_landmark_extraction_failures.csv"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=== EchoGesture: Full-Scale Landmark Extraction (v3 Audited) ===")
    print(f"Manifest source: {manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"Batch checkpoint size: {batch_size}")

    # Read records from normalized manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = [r for r in reader if r["dataset_source"] == "MS-ASL" and r["download_status"] == "downloaded_verified"]

    total_candidates = len(all_records)
    print(f"Total candidate videos to process: {total_candidates}")
    assert total_candidates == 1452, f"Expected 1452 verified records, found {total_candidates}"

    # Initialize MediaPipe extractor (v3 settings)
    extractor = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        static_image_mode=False
    )

    # Initialize tracking structures
    results_map: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    
    # Load previous checkpoint json if exists
    if os.path.exists(checkpoint_json_path):
        try:
            with open(checkpoint_json_path, "r", encoding="utf-8") as f:
                saved_checkpoints = json.load(f)
                results_map = saved_checkpoints.get("results_map", {})
                failures = saved_checkpoints.get("failures", [])
            print(f"Resumed from checkpoint: {len(results_map)} results loaded, {len(failures)} failures recorded.")
        except Exception as e:
            print(f"Warning: Failed to load existing checkpoint: {e}")

    num_batches = (total_candidates + batch_size - 1) // batch_size
    overall_start_time = time.time()

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min((b_idx + 1) * batch_size, total_candidates)
        batch_slice = all_records[b_start:b_end]

        print(f"\n=======================================================")
        print(f"Starting Batch {b_idx + 1}/{num_batches} (Items {b_start + 1} to {b_end} of {total_candidates})")
        print(f"=======================================================")

        batch_start_time = time.time()
        batch_processed = 0
        batch_newly_extracted = 0
        batch_skipped_existing = 0
        batch_failed = 0
        batch_frames = 0

        for r in batch_slice:
            vid_id = r["video_id"]
            vpath = os.path.normpath(r["video_path"])
            npz_name = f"{vid_id}.npz"
            npz_path = os.path.join(output_dir, npz_name)

            # Check if valid file already exists
            if is_valid_npz(npz_path) and vid_id in results_map:
                batch_skipped_existing += 1
                batch_processed += 1
                continue

            try:
                t0 = time.time()
                data_dict, stats = extractor.process_video(vpath)

                # Save compressed npz
                np.savez_compressed(
                    npz_path,
                    norm_landmarks=data_dict["norm_landmarks"],
                    world_landmarks=data_dict["world_landmarks"],
                    presence=data_dict["presence"],
                    handedness_score=data_dict["handedness_score"],
                    confidence=data_dict["confidence"],
                    both_hands_present=data_dict["both_hands_present"],
                    inter_hand_distance_geom=data_dict["inter_hand_distance_geom"],
                    inter_hand_distance=data_dict["inter_hand_distance"],
                    inter_hand_offset_geom=data_dict["inter_hand_offset_geom"],
                    inter_hand_offset_2d=data_dict["inter_hand_offset_2d"],
                    flat_features=data_dict["flat_features"]
                )

                file_size = os.path.getsize(npz_path)
                result_entry = {
                    **r,
                    "npz_file": npz_name,
                    "npz_path": npz_path,
                    "npz_size_bytes": file_size,
                    **stats
                }
                results_map[vid_id] = result_entry
                batch_newly_extracted += 1
                batch_frames += stats["total_frames"]

            except Exception as e:
                batch_failed += 1
                err_msg = str(e)
                print(f"  [FAIL] {vid_id} ({vpath}): {err_msg}")
                fail_entry = {
                    **r,
                    "npz_file": None,
                    "npz_path": None,
                    "npz_size_bytes": 0,
                    "extraction_status": "FAILED",
                    "error_message": err_msg
                }
                failures.append(fail_entry)
                results_map[vid_id] = fail_entry

            batch_processed += 1

        batch_elapsed = time.time() - batch_start_time
        batch_fps = batch_frames / batch_elapsed if batch_elapsed > 0 else 0.0

        # Check total storage & free disk
        total_npz_bytes = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".npz"))
        total_d, used_d, free_d = shutil.disk_usage("C:")

        # Checkpoint summary log entry
        log_line = (
            f"Checkpoint Batch {b_idx + 1}/{num_batches}: "
            f"Processed={batch_processed} (New={batch_newly_extracted}, Skipped={batch_skipped_existing}, Failed={batch_failed}) | "
            f"Frames={batch_frames} ({batch_fps:.1f} FPS) | "
            f"BatchTime={batch_elapsed:.1f}s | "
            f"TotalNPZSize={total_npz_bytes / (1024**2):.2f} MB | "
            f"FreeDisk={free_d / (1024**3):.2f} GB\n"
        )
        print(f"\n--> {log_line.strip()}")

        with open(checkpoint_log_path, "a", encoding="utf-8") as f_log:
            f_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_line}")

        # Save checkpoint JSON
        with open(checkpoint_json_path, "w", encoding="utf-8") as f_chk:
            json.dump({
                "last_completed_batch": b_idx + 1,
                "total_batches": num_batches,
                "total_candidates": total_candidates,
                "results_map": results_map,
                "failures": failures,
                "timestamp": time.time()
            }, f_chk)

    extractor.close()
    overall_elapsed = time.time() - overall_start_time

    # Generate feature_manifest.csv
    manifest_fields = [
        "video_id", "original_label", "normalized_label", "dataset_source",
        "signer_id", "original_split", "unified_split", "video_path",
        "npz_path", "total_frames", "width", "height", "fps", "duration_sec",
        "any_hand_detection_rate", "both_hands_detection_rate",
        "left_hand_detection_rate", "right_hand_detection_rate",
        "mean_handedness_score_left", "mean_handedness_score_right",
        "processing_time_sec", "npz_size_bytes", "extraction_status"
    ]

    with open(feature_manifest_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_records:
            vid_id = r["video_id"]
            if vid_id in results_map:
                writer.writerow(results_map[vid_id])
            else:
                writer.writerow({**r, "extraction_status": "NOT_PROCESSED"})

    # Generate full_landmark_extraction_failures.csv
    with open(failures_csv_path, "w", newline="", encoding="utf-8") as f_fail:
        fail_fields = ["video_id", "normalized_label", "signer_id", "unified_split", "video_path", "error_message"]
        writer = csv.DictWriter(f_fail, fieldnames=fail_fields, extrasaction="ignore")
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)

    print("\n=======================================================")
    print(f"FULL EXTRACTION COMPLETE in {overall_elapsed / 60:.2f} minutes")
    print(f"Total Successful Feature Tensors: {len([r for r in results_map.values() if r.get('extraction_status') == 'SUCCESS'])}")
    print(f"Total Failed Videos: {len(failures)}")
    print(f"Feature Manifest saved to: {feature_manifest_path}")
    print(f"Failures Log saved to: {failures_csv_path}")
    print("=======================================================")


if __name__ == "__main__":
    main()
