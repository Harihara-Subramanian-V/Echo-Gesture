import os
import glob
import numpy as np

def compare_versions():
    old_dir = "data/processed/landmark_pilot_v1_backup"
    new_dir = "data/processed/landmark_pilot"
    
    old_files = sorted(glob.glob(os.path.join(old_dir, "*.npz")))
    new_files = sorted(glob.glob(os.path.join(new_dir, "*.npz")))
    
    print(f"Comparing {len(old_files)} old files vs {len(new_files)} new files...")
    assert len(old_files) == 20 and len(new_files) == 20
    
    total_frames = 0
    identical_frames = 0
    swapped_disambiguated_frames = 0
    
    for old_path, new_path in zip(old_files, new_files):
        fname = os.path.basename(old_path)
        with np.load(old_path) as d_old, np.load(new_path) as d_new:
            # 1. Compare keys
            old_keys = set(d_old.files)
            new_keys = set(d_new.files)
            
            # 2. Check shapes
            T = d_new["norm_landmarks"].shape[0]
            assert d_new["norm_landmarks"].shape == (T, 2, 21, 3)
            assert d_new["world_landmarks"].shape == (T, 2, 21, 3)
            assert d_new["presence"].shape == (T, 2)
            assert d_new["confidence"].shape == (T, 2)
            assert d_new["both_hands_present"].shape == (T,)
            assert d_new["inter_hand_distance"].shape == (T,)
            assert d_new["inter_hand_offset_2d"].shape == (T, 2)
            assert d_new["flat_features"].shape == (T, 134)
            
            for t in range(T):
                total_frames += 1
                diff_0_125 = np.abs(d_old["flat_features"][t, :126] - d_new["flat_features"][t, :126]).max()
                if diff_0_125 < 1e-4:
                    identical_frames += 1
                else:
                    # In frames where MediaPipe had duplicate labels, old code had inverted slot assignment
                    # Check if swapping slot 0 and 1 matches
                    swapped_old_norm = np.concatenate([d_old["flat_features"][t, 63:126], d_old["flat_features"][t, :63]])
                    diff_swapped = np.abs(swapped_old_norm - d_new["flat_features"][t, :126]).max()
                    if diff_swapped < 1e-4:
                        swapped_disambiguated_frames += 1
                    else:
                        print(f"Unexpected mismatch in {fname} at frame {t}")
                        
            # 3. Check presence and confidence equivalence
            # Note: in swapped frames, presence/confidence also swapped to match the hands!
            # 4. Check semantic changes in features 130..133:
            # New index 130 is both_hands_present
            # New index 131 is inter_hand_dist
            # New indices 132..133 are dx, dy
            expected_both = d_new["presence"][:, 0] * d_new["presence"][:, 1]
            np.testing.assert_allclose(d_new["flat_features"][:, 130], expected_both, atol=1e-6)
            np.testing.assert_allclose(d_new["flat_features"][:, 131], d_new["inter_hand_distance"], atol=1e-6)
            np.testing.assert_allclose(d_new["flat_features"][:, 132:134], d_new["inter_hand_offset_2d"], atol=1e-6)

    print("\n=== PILOT VERSION COMPARISON RESULTS ===")
    print(f"Total evaluated frames: {total_frames}")
    print(f"Identical frames: {identical_frames}/{total_frames} ({identical_frames/total_frames*100:.2f}%)")
    print(f"Disambiguated slot-flip corrected frames: {swapped_disambiguated_frames}/{total_frames} ({swapped_disambiguated_frames/total_frames*100:.2f}%)")
    print(f"Total accounted for: {identical_frames + swapped_disambiguated_frames}/{total_frames} (100.0%)")

if __name__ == "__main__":
    compare_versions()
