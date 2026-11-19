"""
EchoGesture Phase 4: Sequence Preprocessing Pipeline
Reproducible temporal resampling and feature standardization for sign language landmarks.

Transforms variable-length 134-D MediaPipe landmark sequences from Phase 3 into
fixed-length, standardized sequence tensors for baseline sequence model training.

Zero Data Leakage: Normalization statistics are computed strictly on the Train partition.
Strict Absence Masking: All absent-hand coordinates are strictly clamped to 0.0.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


def resample_hand_landmarks(
    flat_feats: np.ndarray,
    hand_idx: int,
    target_len: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample a single hand's landmarks and presence temporally.
    
    Args:
        flat_feats: (T, 134) array from Phase 3 extraction
        hand_idx: 0 for Left Hand (0..62), 1 for Right Hand (63..125)
        target_len: Target sequence length (e.g. 64)
        
    Returns:
        resampled_coords: (target_len, 63) float32 coordinates
        resampled_presence: (target_len,) float32 binary presence (0.0 or 1.0)
        resampled_score: (target_len,) float32 handedness score
    """
    T = len(flat_feats)
    coord_start = 0 if hand_idx == 0 else 63
    coord_end = coord_start + 63
    pres_idx = 126 if hand_idx == 0 else 127
    score_idx = 128 if hand_idx == 0 else 129
    
    orig_t = np.arange(T, dtype=np.float32)
    target_t = np.linspace(0.0, float(T - 1), target_len, dtype=np.float32)
    
    pres = flat_feats[:, pres_idx]
    active_mask = (pres > 0.5)
    active_t = orig_t[active_mask]
    
    # Nearest-neighbor presence on target grid
    nearest_idx = np.clip(np.round(target_t).astype(int), 0, T - 1)
    target_pres = (pres[nearest_idx] > 0.5).astype(np.float32)
    
    resampled_coords = np.zeros((target_len, 63), dtype=np.float32)
    resampled_score = np.zeros(target_len, dtype=np.float32)
    
    num_active = len(active_t)
    if num_active == 0:
        # Hand was never detected in this video
        return resampled_coords, target_pres, resampled_score
    elif num_active == 1:
        # Single active detection: propagate to active frames
        single_coords = flat_feats[int(active_t[0]), coord_start:coord_end]
        single_score = flat_feats[int(active_t[0]), score_idx]
        for j in range(target_len):
            if target_pres[j] > 0.5:
                resampled_coords[j] = single_coords
                resampled_score[j] = single_score
        return resampled_coords, target_pres, resampled_score
    else:
        # Multiple active detections: interpolate along active trajectory
        for c in range(63):
            col_vals = flat_feats[active_mask, coord_start + c]
            interp_vals = np.interp(target_t, active_t, col_vals)
            resampled_coords[:, c] = np.where(target_pres > 0.5, interp_vals, 0.0)
            
        score_vals = flat_feats[active_mask, score_idx]
        interp_scores = np.interp(target_t, active_t, score_vals)
        resampled_score[:] = np.where(target_pres > 0.5, interp_scores, 0.0)
        
        return resampled_coords, target_pres, resampled_score


def resample_interaction_features(
    flat_feats: np.ndarray,
    target_pres_left: np.ndarray,
    target_pres_right: np.ndarray,
    target_len: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample two-hand interaction features: both_hands_present, distance, and offset.
    
    Args:
        flat_feats: (T, 134) array
        target_pres_left: (target_len,) resampled left presence
        target_pres_right: (target_len,) resampled right presence
        target_len: Target sequence length
        
    Returns:
        target_both: (target_len,) binary flag
        resampled_dist: (target_len,) float32 distance
        resampled_offset: (target_len, 2) float32 (dx, dy)
    """
    T = len(flat_feats)
    orig_t = np.arange(T, dtype=np.float32)
    target_t = np.linspace(0.0, float(T - 1), target_len, dtype=np.float32)
    
    # Target both-hands flag is strictly both hands present
    target_both = ((target_pres_left > 0.5) & (target_pres_right > 0.5)).astype(np.float32)
    
    both_orig = flat_feats[:, 130] > 0.5
    active_both_t = orig_t[both_orig]
    
    resampled_dist = np.zeros(target_len, dtype=np.float32)
    resampled_offset = np.zeros((target_len, 2), dtype=np.float32)
    
    num_active = len(active_both_t)
    if num_active == 0:
        return target_both, resampled_dist, resampled_offset
    elif num_active == 1:
        s_idx = int(active_both_t[0])
        s_dist = flat_feats[s_idx, 131]
        s_off = flat_feats[s_idx, 132:134]
        for j in range(target_len):
            if target_both[j] > 0.5:
                resampled_dist[j] = s_dist
                resampled_offset[j] = s_off
        return target_both, resampled_dist, resampled_offset
    else:
        dist_vals = flat_feats[both_orig, 131]
        interp_dist = np.interp(target_t, active_both_t, dist_vals)
        resampled_dist[:] = np.where(target_both > 0.5, interp_dist, 0.0)
        
        for d in range(2):
            off_vals = flat_feats[both_orig, 132 + d]
            interp_off = np.interp(target_t, active_both_t, off_vals)
            resampled_offset[:, d] = np.where(target_both > 0.5, interp_off, 0.0)
            
        return target_both, resampled_dist, resampled_offset


def resample_sequence(flat_feats: np.ndarray, target_len: int = 64) -> np.ndarray:
    """
    Resample a full (T, 134) sequence into a (target_len, 134) sequence with strict masking.
    """
    coords_L, pres_L, score_L = resample_hand_landmarks(flat_feats, 0, target_len)
    coords_R, pres_R, score_R = resample_hand_landmarks(flat_feats, 1, target_len)
    both_flag, dist_geom, offset_geom = resample_interaction_features(flat_feats, pres_L, pres_R, target_len)
    
    # Reassemble into exactly 134 dimensions
    resampled_seq = np.zeros((target_len, 134), dtype=np.float32)
    resampled_seq[:, 0:63] = coords_L
    resampled_seq[:, 63:126] = coords_R
    resampled_seq[:, 126] = pres_L
    resampled_seq[:, 127] = pres_R
    resampled_seq[:, 128] = score_L
    resampled_seq[:, 129] = score_R
    resampled_seq[:, 130] = both_flag
    resampled_seq[:, 131] = dist_geom
    resampled_seq[:, 132:134] = offset_geom
    
    return resampled_seq


def fit_scaler_on_train(train_sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute mean and standard deviation strictly on the training partition.
    Only active frames are used for each component to avoid skewing by 0-masks.
    
    Dimension breakdown:
    - 0..62: Left Hand coords -> active where presence_left == 1.0
    - 63..125: Right Hand coords -> active where presence_right == 1.0
    - 126..127: Binary presence flags -> mean=0, std=1 (preserved as {0, 1})
    - 128..129: Handedness scores -> mean=0, std=1 (preserved in [0, 1])
    - 130: both_hands_present flag -> mean=0, std=1 (preserved as {0, 1})
    - 131..133: Interaction features -> active where both_hands_present == 1.0
    """
    all_left = []
    all_right = []
    all_inter = []
    
    for seq in train_sequences:
        p_L = seq[:, 126] > 0.5
        p_R = seq[:, 127] > 0.5
        p_both = seq[:, 130] > 0.5
        
        if np.any(p_L):
            all_left.append(seq[p_L, 0:63])
        if np.any(p_R):
            all_right.append(seq[p_R, 63:126])
        if np.any(p_both):
            all_inter.append(seq[p_both, 131:134])
            
    concat_left = np.vstack(all_left) if all_left else np.zeros((1, 63), dtype=np.float32)
    concat_right = np.vstack(all_right) if all_right else np.zeros((1, 63), dtype=np.float32)
    concat_inter = np.vstack(all_inter) if all_inter else np.zeros((1, 3), dtype=np.float32)
    
    mean_full = np.zeros(134, dtype=np.float32)
    std_full = np.ones(134, dtype=np.float32)
    
    # Left hand coordinates (0..62)
    mean_full[0:63] = np.mean(concat_left, axis=0)
    std_l = np.std(concat_left, axis=0)
    std_l[std_l < 1e-6] = 1.0
    std_full[0:63] = std_l
    
    # Right hand coordinates (63..125)
    mean_full[63:126] = np.mean(concat_right, axis=0)
    std_r = np.std(concat_right, axis=0)
    std_r[std_r < 1e-6] = 1.0
    std_full[63:126] = std_r
    
    # Interaction features (131..133)
    mean_full[131:134] = np.mean(concat_inter, axis=0)
    std_inter = np.std(concat_inter, axis=0)
    std_inter[std_inter < 1e-6] = 1.0
    std_full[131:134] = std_inter
    
    return {
        "mean": mean_full,
        "std": std_full
    }


def apply_scaler(seq: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Standardize a sequence using the fitted scaler, preserving strict zero-masking.
    """
    normalized = seq.copy()
    mean = scaler["mean"]
    std = scaler["std"]
    
    p_L = normalized[:, 126] > 0.5
    p_R = normalized[:, 127] > 0.5
    p_both = normalized[:, 130] > 0.5
    
    # Normalize Left hand active frames
    if np.any(p_L):
        normalized[p_L, 0:63] = (normalized[p_L, 0:63] - mean[0:63]) / std[0:63]
    normalized[~p_L, 0:63] = 0.0
    
    # Normalize Right hand active frames
    if np.any(p_R):
        normalized[p_R, 63:126] = (normalized[p_R, 63:126] - mean[63:126]) / std[63:126]
    normalized[~p_R, 63:126] = 0.0
    
    # Normalize Interaction active frames
    if np.any(p_both):
        normalized[p_both, 131:134] = (normalized[p_both, 131:134] - mean[131:134]) / std[131:134]
    normalized[~p_both, 131:134] = 0.0
    
    return normalized


def run_pipeline(manifest_path: str, output_dir: str, target_len: int = 64):
    """Execute complete sequence preprocessing pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== EchoGesture Sequence Preprocessing Pipeline ===")
    print(f"Reading manifest: {manifest_path}")
    print(f"Target sequence length: {target_len}")
    print(f"Output directory: {output_dir}")
    
    df = pd.read_csv(manifest_path)
    total_clips = len(df)
    print(f"Total clips in manifest: {total_clips}")
    
    # Build Label Encoder (alphabetical 40 classes)
    unique_labels = sorted(df["normalized_label"].unique())
    num_classes = len(unique_labels)
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx2label = {i: lbl for i, lbl in enumerate(unique_labels)}
    
    print(f"Unique classes: {num_classes}")
    assert num_classes == 40, f"Expected 40 classes, found {num_classes}"
    
    # Save label encoder
    encoder_path = out_path / "label_encoder.json"
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump({
            "label2idx": label2idx,
            "idx2label": idx2label,
            "num_classes": num_classes,
            "classes": unique_labels
        }, f, indent=2)
    print(f"Saved label encoder to {encoder_path}")
    
    # Load and resample all sequences per split
    split_data = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    split_ids = {"train": [], "val": [], "test": []}
    
    print("Resampling all sequences...")
    for idx, row in df.iterrows():
        video_id = row["video_id"]
        split = row["unified_split"]
        label = row["normalized_label"]
        label_id = label2idx[label]
        npz_file = row["npz_path"]
        
        # Load raw Phase 3 npz
        data = np.load(npz_file)
        flat_feats = data["flat_features"]
        
        # Resample to fixed length
        resampled = resample_sequence(flat_feats, target_len=target_len)
        
        split_data[split].append(resampled)
        split_labels[split].append(label_id)
        split_ids[split].append(video_id)
        
        if (idx + 1) % 250 == 0 or (idx + 1) == total_clips:
            print(f"  Processed {idx + 1}/{total_clips} sequences...")
            
    print("\nResampling complete:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(split_data[split])} sequences")
        
    # Fit scaler strictly on Train
    print("\nFitting scaler strictly on Train partition (Zero Data Leakage)...")
    scaler = fit_scaler_on_train(split_data["train"])
    
    # Save scaler parameters
    scaler_path = out_path / "scaler_params.npz"
    np.savez_compressed(
        scaler_path,
        mean=scaler["mean"],
        std=scaler["std"]
    )
    print(f"Saved scaler parameters to {scaler_path}")
    
    # Apply scaler to all partitions
    print("Applying standardization with strict zero-masking preservation...")
    processed_arrays = {}
    for split in ["train", "val", "test"]:
        X_split = np.array([apply_scaler(seq, scaler) for seq in split_data[split]], dtype=np.float32)
        y_split = np.array(split_labels[split], dtype=np.int32)
        ids_split = np.array(split_ids[split], dtype=str)
        
        # Integrity checks
        assert not np.isnan(X_split).any(), f"NaN detected in {split} features!"
        assert not np.isinf(X_split).any(), f"Inf detected in {split} features!"
        assert X_split.shape == (len(split_data[split]), target_len, 134), f"Shape mismatch in {split}: {X_split.shape}"
        
        # Verify strict zero-masking
        p_L = X_split[:, :, 126] == 0.0
        p_R = X_split[:, :, 127] == 0.0
        assert np.all(X_split[:, :, 0:63][p_L] == 0.0), f"Zero-masking violation in {split} Left Hand!"
        assert np.all(X_split[:, :, 63:126][p_R] == 0.0), f"Zero-masking violation in {split} Right Hand!"
        
        save_file = out_path / f"{split}_sequences.npz"
        np.savez_compressed(
            save_file,
            X=X_split,
            y=y_split,
            video_ids=ids_split
        )
        print(f"  Saved {split} sequences: {save_file} (X shape: {X_split.shape}, y shape: {y_split.shape})")
        processed_arrays[split] = (X_split, y_split, ids_split)
        
    print("\nAll sequences preprocessed and saved successfully!")
    return processed_arrays, unique_labels, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Sequence Preprocessing")
    parser.add_argument("--manifest", type=str, default="data/processed/landmarks_v3/feature_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--target_len", type=int, default=64)
    args = parser.parse_args()
    
    run_pipeline(args.manifest, args.output_dir, args.target_len)
