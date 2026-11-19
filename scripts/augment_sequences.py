"""
EchoGesture Phase 5A: Sequence Augmentation & Feature Engineering Module
Provides dynamic training-only data augmentation and temporal velocity feature extraction.

Guarantees:
- Strict zero-masking preservation: absent hands remain strictly 0.0
- No velocity spikes across hand appearance/disappearance transitions
- Anatomically valid spatial scaling and smooth temporal speed warping
- Dynamic on-the-fly augmentation during training (zero data leakage)
"""

import math
from typing import Tuple, Optional
import numpy as np
import keras


def compute_velocity_features(seq_134: np.ndarray) -> np.ndarray:
    """
    Calculate first-order temporal differences for landmark coordinates and interactions.
    
    Spike Prevention Rule:
    Velocity is non-zero ONLY when the hand is present in BOTH frame t and frame t-1.
    If a hand appears, disappears, or is absent, velocity is strictly 0.0.
    
    Args:
        seq_134: (T, 134) float32 landmark sequence
        
    Returns:
        seq_263: (T, 263) float32 array:
            - [0:134]: Original 134-D features
            - [134:197]: Left Hand landmark velocities (63)
            - [197:260]: Right Hand landmark velocities (63)
            - [260:263]: Two-hand interaction velocities (3: distance, dx, dy)
    """
    T, D = seq_134.shape
    assert D == 134, f"Expected 134 features, got {D}"
    
    # 1. Left Hand Velocity (indices 0..62, presence index 126)
    p_L = seq_134[:, 126] > 0.5
    valid_L = np.zeros(T, dtype=bool)
    valid_L[1:] = p_L[1:] & p_L[:-1]  # Present in both t and t-1
    v_L = np.zeros((T, 63), dtype=np.float32)
    v_L[1:] = np.where(valid_L[1:, None], seq_134[1:, 0:63] - seq_134[:-1, 0:63], 0.0)
    
    # 2. Right Hand Velocity (indices 63..125, presence index 127)
    p_R = seq_134[:, 127] > 0.5
    valid_R = np.zeros(T, dtype=bool)
    valid_R[1:] = p_R[1:] & p_R[:-1]  # Present in both t and t-1
    v_R = np.zeros((T, 63), dtype=np.float32)
    v_R[1:] = np.where(valid_R[1:, None], seq_134[1:, 63:126] - seq_134[:-1, 63:126], 0.0)
    
    # 3. Inter-Hand Interaction Velocity (indices 131..133, co-presence index 130)
    p_both = seq_134[:, 130] > 0.5
    valid_both = np.zeros(T, dtype=bool)
    valid_both[1:] = p_both[1:] & p_both[:-1]  # Co-present in both t and t-1
    v_inter = np.zeros((T, 3), dtype=np.float32)
    v_inter[1:] = np.where(valid_both[1:, None], seq_134[1:, 131:134] - seq_134[:-1, 131:134], 0.0)
    
    return np.concatenate([seq_134, v_L, v_R, v_inter], axis=-1).astype(np.float32)


def batch_compute_velocity(X_batch: np.ndarray) -> np.ndarray:
    """Compute velocity features across an entire batch of sequences."""
    return np.array([compute_velocity_features(seq) for seq in X_batch], dtype=np.float32)


def apply_temporal_speed_warp(seq: np.ndarray, speed_min: float = 0.85, speed_max: float = 1.15) -> np.ndarray:
    """
    Simulate signing speed variation (0.85x - 1.15x) via smooth temporal window resampling.
    """
    T, D = seq.shape
    warped = seq.copy()
    
    # Random window duration
    speed = np.random.uniform(speed_min, speed_max)
    eff_len = T / speed
    
    if eff_len < T:
        # Faster sign: occupies a sub-window, randomly offset within sequence
        max_start = T - eff_len
        t0 = np.random.uniform(0.0, max_start)
        t1 = t0 + eff_len
    else:
        # Slower sign: stretches across boundaries slightly
        overflow = (eff_len - T) / 2.0
        t0 = max(0.0, -overflow * np.random.uniform(0.0, 1.0))
        t1 = min(float(T - 1), float(T - 1) + overflow * np.random.uniform(0.0, 1.0))
        
    query_t = np.linspace(t0, t1, T)
    nearest_idx = np.clip(np.round(query_t).astype(int), 0, T - 1)
    
    # Resample presence flags using nearest neighbor
    p_L = seq[nearest_idx, 126]
    p_R = seq[nearest_idx, 127]
    
    # Resample continuous channels
    for c in range(126):
        warped[:, c] = np.interp(query_t, np.arange(T), seq[:, c])
    warped[:, 126] = p_L
    warped[:, 127] = p_R
    warped[:, 128] = np.where(p_L > 0.5, np.interp(query_t, np.arange(T), seq[:, 128]), 0.0)
    warped[:, 129] = np.where(p_R > 0.5, np.interp(query_t, np.arange(T), seq[:, 129]), 0.0)
    
    both_present = ((p_L > 0.5) & (p_R > 0.5)).astype(np.float32)
    warped[:, 130] = both_present
    for c in range(131, 134):
        interp_c = np.interp(query_t, np.arange(T), seq[:, c])
        warped[:, c] = np.where(both_present > 0.5, interp_c, 0.0)
        
    # Re-enforce zero masks
    warped[p_L <= 0.5, 0:63] = 0.0
    warped[p_R <= 0.5, 63:126] = 0.0
    
    return warped


def apply_spatial_scale_and_jitter(
    seq: np.ndarray,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    noise_sigma: float = 0.005
) -> np.ndarray:
    """
    Apply anatomically valid spatial scaling (+-10%) and slight coordinate sensor jitter.
    Only active frames receive perturbation; absent frames remain strictly 0.0.
    """
    aug = seq.copy()
    
    scale_L = np.random.uniform(scale_range[0], scale_range[1])
    scale_R = np.random.uniform(scale_range[0], scale_range[1])
    
    p_L = aug[:, 126] > 0.5
    p_R = aug[:, 127] > 0.5
    p_both = aug[:, 130] > 0.5
    
    if np.any(p_L):
        coords_L = aug[p_L, 0:63]
        noise_L = np.random.normal(0, noise_sigma, size=coords_L.shape).astype(np.float32)
        aug[p_L, 0:63] = coords_L * scale_L + noise_L
    aug[~p_L, 0:63] = 0.0
    
    if np.any(p_R):
        coords_R = aug[p_R, 63:126]
        noise_R = np.random.normal(0, noise_sigma, size=coords_R.shape).astype(np.float32)
        aug[p_R, 63:126] = coords_R * scale_R + noise_R
    aug[~p_R, 63:126] = 0.0
    
    aug[~p_both, 131:134] = 0.0
    return aug


def augment_sequence_134(seq_134: np.ndarray, p_aug: float = 0.6) -> np.ndarray:
    """
    Full dynamic augmentation pipeline for a single 134-D sequence.
    """
    if np.random.rand() > p_aug:
        return seq_134.copy()
        
    aug = seq_134.copy()
    
    # 50% chance of temporal speed variation
    if np.random.rand() < 0.5:
        aug = apply_temporal_speed_warp(aug, speed_min=0.85, speed_max=1.15)
        
    # Spatial scaling and sensor noise
    aug = apply_spatial_scale_and_jitter(aug, scale_range=(0.90, 1.10), noise_sigma=0.005)
    return aug


class DynamicAugmentedSequence(keras.utils.PyDataset):
    """
    Thread-safe Keras PyDataset for on-the-fly training data augmentation.
    Generates fresh augmented sequences at each epoch without touching disk or validation data.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augment: bool = True,
        use_velocity: bool = False,
        shuffle: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.use_velocity = use_velocity
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_X = self.X[batch_idx].copy()
        batch_y = self.y[batch_idx]
        
        # 1. Apply spatial/temporal augmentation if enabled
        if self.augment:
            batch_X = np.array([augment_sequence_134(seq, p_aug=0.7) for seq in batch_X], dtype=np.float32)
            
        # 2. Compute velocity features if enabled
        if self.use_velocity:
            batch_X = batch_compute_velocity(batch_X)
            
        return batch_X, batch_y
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
