import os
import glob
import json
import csv
import numpy as np

def audit_results():
    pilot_dir = "data/processed/landmark_pilot"
    manifest_path = os.path.join(pilot_dir, "pilot_manifest.json")
    summary_path = os.path.join(pilot_dir, "pilot_summary.csv")
    
    assert os.path.exists(manifest_path), "pilot_manifest.json missing"
    assert os.path.exists(summary_path), "pilot_summary.csv missing"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"Total records in pilot manifest: {len(records)}")
    
    npz_files = glob.glob(os.path.join(pilot_dir, "*.npz"))
    print(f"Total .npz files on disk: {len(npz_files)}")
    assert len(npz_files) == 20, f"Expected 20 .npz files, found {len(npz_files)}"
    
    # Audit each .npz file
    total_frames = 0
    total_any_detected = 0
    total_both_detected = 0
    total_size_bytes = 0
    
    one_handed_stats = {"total_frames": 0, "any_hand": 0, "both_hands": 0}
    two_handed_stats = {"total_frames": 0, "any_hand": 0, "both_hands": 0}
    
    for r in records:
        npz_path = r["npz_path"]
        assert os.path.exists(npz_path), f"Missing npz file: {npz_path}"
        total_size_bytes += os.path.getsize(npz_path)
        
        with np.load(npz_path) as data:
            norm_lms = data["norm_landmarks"]
            world_lms = data["world_landmarks"]
            presence = data["presence"]
            confidence = data["confidence"]
            dist = data["inter_hand_distance"]
            offset = data["inter_hand_offset"]
            flat = data["flat_features"]
            
            T = r["total_frames"]
            assert norm_lms.shape == (T, 2, 21, 3), f"Bad norm_lms shape: {norm_lms.shape}"
            assert world_lms.shape == (T, 2, 21, 3), f"Bad world_lms shape: {world_lms.shape}"
            assert presence.shape == (T, 2), f"Bad presence shape: {presence.shape}"
            assert confidence.shape == (T, 2), f"Bad confidence shape: {confidence.shape}"
            assert dist.shape == (T,), f"Bad dist shape: {dist.shape}"
            assert offset.shape == (T, 3), f"Bad offset shape: {offset.shape}"
            assert flat.shape == (T, 134), f"Bad flat shape: {flat.shape}"
            
            # Check value constraints
            assert not np.isnan(flat).any(), "NaN in flat_features!"
            assert not np.isinf(flat).any(), "Inf in flat_features!"
            
            # Check presence aligns with non-zero coordinates
            for t in range(T):
                for h in range(2):
                    if presence[t, h] == 0.0:
                        assert np.all(norm_lms[t, h] == 0.0), f"Non-zero norm coordinates for absent hand at frame {t}, hand {h}"
                        assert np.all(world_lms[t, h] == 0.0), f"Non-zero world coordinates for absent hand at frame {t}, hand {h}"
                    else:
                        assert np.any(norm_lms[t, h] != 0.0), f"Zero coordinates for present hand at frame {t}, hand {h}"
                        
                # Check two-handed distance
                if presence[t, 0] == 1.0 and presence[t, 1] == 1.0:
                    expected_dist = np.linalg.norm(norm_lms[t, 1, 0, :2] - norm_lms[t, 0, 0, :2])
                    assert np.isclose(dist[t], expected_dist, atol=1e-5), f"Distance mismatch at frame {t}"
                else:
                    assert dist[t] == 0.0, f"Distance non-zero when both hands not present at frame {t}"
                    assert np.all(offset[t] == 0.0), f"Offset non-zero when both hands not present at frame {t}"
                    
        total_frames += r["total_frames"]
        total_any_detected += r["frames_with_any_hand"]
        total_both_detected += r["frames_with_both_hands"]
        
        htype = r["handedness_type"]
        if htype == "1-handed":
            one_handed_stats["total_frames"] += r["total_frames"]
            one_handed_stats["any_hand"] += r["frames_with_any_hand"]
            one_handed_stats["both_hands"] += r["frames_with_both_hands"]
        else:
            two_handed_stats["total_frames"] += r["total_frames"]
            two_handed_stats["any_hand"] += r["frames_with_any_hand"]
            two_handed_stats["both_hands"] += r["frames_with_both_hands"]
            
    print("\n=== DATA INTEGRITY AUDIT: 100% PASS ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Total .npz storage: {total_size_bytes} bytes ({total_size_bytes / 1024:.2f} KB, avg {total_size_bytes / 20 / 1024:.2f} KB/clip)")
    print(f"Overall any-hand detection rate: {total_any_detected}/{total_frames} ({total_any_detected/total_frames*100:.2f}%)")
    print(f"Overall both-hands detection rate: {total_both_detected}/{total_frames} ({total_both_detected/total_frames*100:.2f}%)")
    
    print("\n--- 1-Handed Signs (10 clips) ---")
    oh_tot = one_handed_stats["total_frames"]
    oh_any = one_handed_stats["any_hand"]
    oh_both = one_handed_stats["both_hands"]
    print(f"Frames: {oh_tot}, Any-hand: {oh_any} ({oh_any/oh_tot*100:.2f}%), Both-hands: {oh_both} ({oh_both/oh_tot*100:.2f}%)")
    
    print("\n--- 2-Handed Signs (10 clips) ---")
    th_tot = two_handed_stats["total_frames"]
    th_any = two_handed_stats["any_hand"]
    th_both = two_handed_stats["both_hands"]
    print(f"Frames: {th_tot}, Any-hand: {th_any} ({th_any/th_tot*100:.2f}%), Both-hands: {th_both} ({th_both/th_tot*100:.2f}%)")

if __name__ == "__main__":
    audit_results()
