import os
import glob
import numpy as np

def audit_temporal_consistency():
    pilot_dir = "data/processed/landmark_pilot"
    files = sorted(glob.glob(os.path.join(pilot_dir, "*.npz")))
    
    print(f"Auditing temporal slot consistency across {len(files)} videos...")
    
    total_two_hand_frames = 0
    slot_jumps = 0
    large_displacement_count = 0
    
    for f in files:
        fname = os.path.basename(f)
        d = np.load(f)
        norm = d["norm_landmarks"]  # (T, 2, 21, 3)
        pres = d["presence"]        # (T, 2)
        T = norm.shape[0]
        
        # Check wrist positions over time
        for t in range(1, T):
            # If both hands present in both t-1 and t
            if pres[t-1, 0] == 1.0 and pres[t-1, 1] == 1.0 and pres[t, 0] == 1.0 and pres[t, 1] == 1.0:
                total_two_hand_frames += 1
                
                # Normal displacement
                d0 = np.linalg.norm(norm[t, 0, 0, :2] - norm[t-1, 0, 0, :2])
                d1 = np.linalg.norm(norm[t, 1, 0, :2] - norm[t-1, 1, 0, :2])
                
                # Swapped displacement
                d0_swap = np.linalg.norm(norm[t, 1, 0, :2] - norm[t-1, 0, 0, :2])
                d1_swap = np.linalg.norm(norm[t, 0, 0, :2] - norm[t-1, 1, 0, :2])
                
                # If swapped displacement is much smaller than normal displacement, a slot flip occurred!
                if (d0_swap + d1_swap) < (d0 + d1) * 0.6 and (d0 > 0.15 or d1 > 0.15):
                    slot_jumps += 1
                    print(f"Potential slot jump in {fname} at frame {t}:")
                    print(f"  Normal dist: {d0:.4f} + {d1:.4f} = {d0+d1:.4f}")
                    print(f"  Swapped dist: {d0_swap:.4f} + {d1_swap:.4f} = {d0_swap+d1_swap:.4f}")
                    print(f"  Prev wrists: L={norm[t-1, 0, 0, :2]}, R={norm[t-1, 1, 0, :2]}")
                    print(f"  Curr wrists: L={norm[t, 0, 0, :2]}, R={norm[t, 1, 0, :2]}")
                    
    print(f"\nTotal two-hand consecutive frames audited: {total_two_hand_frames}")
    print(f"Detected slot jumps: {slot_jumps}")

if __name__ == "__main__":
    audit_temporal_consistency()
