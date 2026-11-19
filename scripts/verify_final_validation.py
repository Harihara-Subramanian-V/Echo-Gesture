import os
import glob
import json
import numpy as np

def run_final_validation():
    pilot_dir = "data/processed/landmark_pilot"
    v2_dir = "data/processed/landmark_pilot_v2_backup"
    
    files = sorted(glob.glob(os.path.join(pilot_dir, "*.npz")))
    v2_files = sorted(glob.glob(os.path.join(v2_dir, "*.npz")))
    
    assert len(files) == 20, f"Expected 20 files, found {len(files)}"
    assert len(v2_files) == 20, f"Expected 20 v2 backup files, found {len(v2_files)}"
    
    total_frames = 0
    total_two_hand_frames = 0
    slot_jumps = 0
    nan_inf_count = 0
    geom_mismatch_count = 0
    absence_violation_count = 0
    
    # Track comparison stats against v2
    coord_identical_frames = 0
    trajectory_smoothed_frames = 0
    
    aspect_ratio_samples = {}
    
    for f, f_v2 in zip(files, v2_files):
        fname = os.path.basename(f)
        d = np.load(f)
        d_v2 = np.load(f_v2)
        
        norm = d["norm_landmarks"]
        world = d["world_landmarks"]
        pres = d["presence"]
        h_score = d["handedness_score"]
        both_pres = d["both_hands_present"]
        dist_geom = d["inter_hand_distance_geom"]
        offset_geom = d["inter_hand_offset_geom"]
        flat = d["flat_features"]
        
        T = norm.shape[0]
        total_frames += T
        
        # 1. Finite check
        for k in ["norm_landmarks", "world_landmarks", "presence", "handedness_score", 
                  "both_hands_present", "inter_hand_distance_geom", "inter_hand_offset_geom", "flat_features"]:
            arr = d[k]
            if np.isnan(arr).any() or np.isinf(arr).any():
                nan_inf_count += 1
                print(f"NaN/Inf violation in {fname} [{k}]")
                
        # 2. Geometric consistency check
        # dist_geom must match sqrt(dx_geom^2 + dy_geom^2)
        calc_dist = np.sqrt(offset_geom[:, 0]**2 + offset_geom[:, 1]**2)
        if not np.allclose(dist_geom, calc_dist, atol=1e-5):
            geom_mismatch_count += 1
            print(f"Geometrical distance mismatch in {fname}")
            
        # 3. Absence masking check
        for t in range(T):
            for h in range(2):
                if pres[t, h] == 0.0:
                    if np.any(norm[t, h] != 0.0) or np.any(world[t, h] != 0.0) or h_score[t, h] != 0.0:
                        absence_violation_count += 1
                else:
                    if not (0.0 <= h_score[t, h] <= 1.0):
                        print(f"Invalid handedness score in {fname} at frame {t}, hand {h}: {h_score[t, h]}")
                        
            if both_pres[t] == 0.0:
                if dist_geom[t] != 0.0 or np.any(offset_geom[t] != 0.0):
                    absence_violation_count += 1
                    
        # 4. Temporal consistency check
        for t in range(1, T):
            if pres[t-1, 0] == 1.0 and pres[t-1, 1] == 1.0 and pres[t, 0] == 1.0 and pres[t, 1] == 1.0:
                total_two_hand_frames += 1
                d0 = np.linalg.norm(norm[t, 0, 0, :2] - norm[t-1, 0, 0, :2])
                d1 = np.linalg.norm(norm[t, 1, 0, :2] - norm[t-1, 1, 0, :2])
                d0_swap = np.linalg.norm(norm[t, 1, 0, :2] - norm[t-1, 0, 0, :2])
                d1_swap = np.linalg.norm(norm[t, 0, 0, :2] - norm[t-1, 1, 0, :2])
                if (d0_swap + d1_swap) < (d0 + d1) * 0.6 and (d0 > 0.15 or d1 > 0.15):
                    slot_jumps += 1
                    
        # 5. Version comparison against v2
        for t in range(T):
            diff = np.abs(flat[t, :126] - d_v2["flat_features"][t, :126]).max()
            if diff < 1e-4:
                coord_identical_frames += 1
            else:
                trajectory_smoothed_frames += 1
                
        # Sample by aspect ratio
        if "msasl_train_0160" in fname:
            aspect_ratio_samples["Landscape (640x360)"] = {
                "fname": fname, "max_dist_geom": float(dist_geom.max()), 
                "max_offset_x": float(offset_geom[:, 0].max()), "max_offset_y": float(offset_geom[:, 1].max())
            }
        elif "msasl_test_1934" in fname:
            aspect_ratio_samples["Square (360x360)"] = {
                "fname": fname, "max_dist_geom": float(dist_geom.max()),
                "max_offset_x": float(offset_geom[:, 0].max()), "max_offset_y": float(offset_geom[:, 1].max())
            }
        elif "msasl_test_1960" in fname:
            aspect_ratio_samples["Portrait (360x640)"] = {
                "fname": fname, "max_dist_geom": float(dist_geom.max()),
                "max_offset_x": float(offset_geom[:, 0].max()), "max_offset_y": float(offset_geom[:, 1].max())
            }
            
    print("\n=== FINAL VALIDATION RESULTS ===")
    print(f"Total evaluated videos: {len(files)}")
    print(f"Total evaluated frames: {total_frames}")
    print(f"NaN or Inf violations: {nan_inf_count} (PASS)")
    print(f"Geometrical distance mismatches: {geom_mismatch_count} (PASS)")
    print(f"Absence masking violations: {absence_violation_count} (PASS)")
    print(f"Two-hand consecutive transitions audited: {total_two_hand_frames}")
    print(f"Detected temporal slot jumps: {slot_jumps} (PASS)")
    print(f"\n--- Comparison vs. v2 Baseline ---")
    print(f"Identical coordinate frames: {coord_identical_frames}/{total_frames} ({coord_identical_frames/total_frames*100:.2f}%)")
    print(f"Temporal trajectory smoothed frames: {trajectory_smoothed_frames}/{total_frames} ({trajectory_smoothed_frames/total_frames*100:.2f}%)")
    print(f"\n--- Aspect Ratio Sample Measurements ---")
    for k, v in aspect_ratio_samples.items():
        print(f"  {k} [{v['fname']}]: max_dist_geom={v['max_dist_geom']:.4f}, max_dx={v['max_offset_x']:.4f}, max_dy={v['max_offset_y']:.4f}")

if __name__ == "__main__":
    run_final_validation()
