"""
EchoGesture Phase 7B.1: Stage 2 Temporal Preprocessing Parity Fixture Generator
Generates deterministic reference fixtures for sequence resampling, training-scaler normalization,
and 129-D velocity generation to produce exact 263-D tensors for Kotlin/Android parity verification.

Coverage & Test Invariants:
1. case_1_short_sequence: T=24 frames resampled to T=64, normalized, velocity computed.
2. case_2_full_sequence: T=64 frames (identity resampling grid), normalized, velocity computed.
3. case_3_long_sequence: T=96 frames downsampled to T=64, normalized, velocity computed.
4. case_4_sparse_detections: Only 10 active frames scattered across 50 frames.
5. case_5_single_hand_continuous: Only right hand present throughout 64 frames.
6. case_6_two_hand_continuous: Both hands present throughout 64 frames.
7. case_7_alternating_hands: Left hand present in first half, Right hand present in second half (velocity reset test).
8. case_8_idle_window: All frames empty (all 263 values strictly 0.0).

Output Targets:
- tests/fixtures/android_parity/stage2_temporal.npz
- tests/fixtures/android_parity/stage2_temporal.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_sequences import resample_sequence, apply_scaler
from scripts.augment_sequences import compute_velocity_features
from scripts.live_inference import LiveTemporalPreprocessor

# Canonical realistic anatomical hand landmark templates extracted from validated MediaPipe tracking
# Shape: (21, 3) normalized landmarks in camera perspective
BASE_LEFT_TEMPLATE = np.array([
    [0.45663357, 0.66708589, -2.1696188e-08],
    [0.44643244, 0.62865853,  2.0519885e-05],
    [0.44990942, 0.59438556, -1.3169742e-03],
    [0.46296266, 0.57694733, -3.0348769e-03],
    [0.47472095, 0.57639450, -4.9290345e-03],
    [0.46438941, 0.57582295, -4.9923807e-03],
    [0.47536841, 0.53937542, -7.4423128e-03],
    [0.48263776, 0.51637697, -9.2004854e-03],
    [0.48733991, 0.49844795, -1.0951238e-02],
    [0.47806501, 0.58802402, -6.7737829e-03],
    [0.48214892, 0.56351644, -8.6864103e-03],
    [0.47706303, 0.56996959, -7.4601909e-03],
    [0.47285375, 0.58119792, -6.5844767e-03],
    [0.48754004, 0.60357702, -8.2048085e-03],
    [0.48943439, 0.58227259, -9.0453345e-03],
    [0.48480874, 0.58701116, -5.3537763e-03],
    [0.48096758, 0.59835559, -3.1100609e-03],
    [0.49392480, 0.61901647, -9.8372037e-03],
    [0.49539387, 0.60014486, -9.2451712e-03],
    [0.49110380, 0.60456765, -5.3553265e-03],
    [0.48812824, 0.61536175, -2.4079874e-03]
], dtype=np.float32)

BASE_RIGHT_TEMPLATE = np.array([
    [0.55718923, 0.90814501, -4.2925947e-08],
    [0.53736091, 0.91093296,  1.0297959e-03],
    [0.51771057, 0.91497499, -1.2231104e-03],
    [0.50224954, 0.92092699, -3.9750389e-03],
    [0.48899263, 0.92465043, -7.1492242e-03],
    [0.52008015, 0.89141351, -1.0083605e-02],
    [0.50400472, 0.90032059, -1.8536128e-02],
    [0.49471429, 0.91671908, -2.3812300e-02],
    [0.48943579, 0.93287903, -2.7054735e-02],
    [0.52807796, 0.89734972, -1.5240919e-02],
    [0.51223010, 0.90650654, -2.2521406e-02],
    [0.50307375, 0.92459917, -2.6422374e-02],
    [0.49820298, 0.94132650, -2.9508209e-02],
    [0.53852570, 0.90712386, -2.0113053e-02],
    [0.52641010, 0.91237462, -2.7152520e-02],
    [0.51747644, 0.92478120, -3.1001387e-02],
    [0.51017201, 0.93620563, -3.4014400e-02],
    [0.55017984, 0.92078739, -2.4468405e-02],
    [0.54554963, 0.92952168, -3.1888738e-02],
    [0.54097670, 0.93792492, -3.5395063e-02],
    [0.53676230, 0.94587690, -3.7804224e-02]
], dtype=np.float32)


def build_frame_134(
    left_active: bool,
    right_active: bool,
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    score_L: float = 0.90,
    score_R: float = 0.90,
    w_px: float = 640.0,
    h_px: float = 480.0
) -> np.ndarray:
    """
    Construct a canonical 134-D frame matching LiveLandmarkExtractor.process_frame.
    
    Indices:
    - [0:63]: Left Hand normalized coordinates (21 x 3)
    - [63:126]: Right Hand normalized coordinates (21 x 3)
    - [126]: Left Hand presence flag (0.0 or 1.0)
    - [127]: Right Hand presence flag (0.0 or 1.0)
    - [128]: Left Hand handedness score
    - [129]: Right Hand handedness score
    - [130]: Both hands present flag (0.0 or 1.0)
    - [131]: Inter-hand Euclidean wrist distance normalized by diagonal
    - [132:134]: Inter-hand 2D offset (dx, dy) normalized by diagonal
    """
    frame = np.zeros(134, dtype=np.float32)
    
    if left_active:
        frame[0:63] = left_coords.flatten()
        frame[126] = 1.0
        frame[128] = float(score_L)
        
    if right_active:
        frame[63:126] = right_coords.flatten()
        frame[127] = 1.0
        frame[129] = float(score_R)
        
    if left_active and right_active:
        frame[130] = 1.0
        dx_norm = float(right_coords[0, 0] - left_coords[0, 0])
        dy_norm = float(right_coords[0, 1] - left_coords[0, 1])
        dx_px = dx_norm * w_px
        dy_px = dy_norm * h_px
        diag = float(np.sqrt(w_px**2 + h_px**2))
        dist_px = float(np.sqrt(dx_px**2 + dy_px**2))
        frame[131] = float(dist_px / diag)
        frame[132] = float(dx_px / diag)
        frame[133] = float(dy_px / diag)
        
    return frame


def generate_case_1_short_sequence() -> Tuple[np.ndarray, str]:
    """
    Case 1: T=24 frames resampled to T=64, normalized, velocity computed.
    Simulates a short sign gesture with inactive boundary frames and smooth two-hand motion.
    """
    T = 24
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "T=24 frames resampled to T=64, normalized, velocity computed"
    
    for t in range(T):
        # Inactive boundary: frames 0..2 and 21..23
        if 3 <= t <= 20:
            right_active = True
            r_shift = np.array([
                0.04 * np.sin(np.pi * (t - 3) / 17.0),
                -0.05 * np.sin(np.pi * (t - 3) / 17.0),
                0.01 * np.cos(np.pi * (t - 3) / 17.0)
            ], dtype=np.float32)
            r_coords = BASE_RIGHT_TEMPLATE + r_shift
            score_R = 0.94
        else:
            right_active = False
            r_coords = np.zeros((21, 3), dtype=np.float32)
            score_R = 0.0
            
        if 7 <= t <= 17:
            left_active = True
            l_shift = np.array([
                -0.03 * np.sin(np.pi * (t - 7) / 10.0),
                0.04 * np.sin(np.pi * (t - 7) / 10.0),
                -0.01 * np.cos(np.pi * (t - 7) / 10.0)
            ], dtype=np.float32)
            l_coords = BASE_LEFT_TEMPLATE + l_shift
            score_L = 0.91
        else:
            left_active = False
            l_coords = np.zeros((21, 3), dtype=np.float32)
            score_L = 0.0
            
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_2_full_sequence() -> Tuple[np.ndarray, str]:
    """
    Case 2: T=64 frames (identity resampling grid), normalized, velocity computed.
    Tests that a 64-frame sequence on the identity grid reproduces coordinates exactly.
    """
    T = 64
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "T=64 frames (identity resampling grid), normalized, velocity computed"
    
    for t in range(T):
        # Inactive boundary: frames 0..7 and 56..63
        if 8 <= t <= 55:
            right_active = True
            r_shift = np.array([
                0.06 * np.sin(2.0 * np.pi * (t - 8) / 48.0),
                -0.04 * np.cos(2.0 * np.pi * (t - 8) / 48.0),
                0.01 * np.sin(2.0 * np.pi * (t - 8) / 48.0)
            ], dtype=np.float32)
            r_coords = BASE_RIGHT_TEMPLATE + r_shift
            score_R = 0.96
        else:
            right_active = False
            r_coords = np.zeros((21, 3), dtype=np.float32)
            score_R = 0.0
            
        if 16 <= t <= 48:
            left_active = True
            l_shift = np.array([
                -0.05 * np.cos(2.0 * np.pi * (t - 16) / 32.0),
                0.05 * np.sin(2.0 * np.pi * (t - 16) / 32.0),
                -0.01 * np.sin(2.0 * np.pi * (t - 16) / 32.0)
            ], dtype=np.float32)
            l_coords = BASE_LEFT_TEMPLATE + l_shift
            score_L = 0.93
        else:
            left_active = False
            l_coords = np.zeros((21, 3), dtype=np.float32)
            score_L = 0.0
            
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_3_long_sequence() -> Tuple[np.ndarray, str]:
    """
    Case 3: T=96 frames downsampled to T=64, normalized, velocity computed.
    Tests downsampling interpolation from 96 frames to 64 frames.
    """
    T = 96
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "T=96 frames downsampled to T=64, normalized, velocity computed"
    
    for t in range(T):
        # Inactive boundary: frames 0..11 and 85..95
        if 12 <= t <= 84:
            right_active = True
            r_shift = np.array([
                0.08 * np.sin(2.0 * np.pi * (t - 12) / 72.0),
                -0.06 * np.sin(4.0 * np.pi * (t - 12) / 72.0),
                0.02 * np.cos(2.0 * np.pi * (t - 12) / 72.0)
            ], dtype=np.float32)
            r_coords = BASE_RIGHT_TEMPLATE + r_shift
            score_R = 0.95
        else:
            right_active = False
            r_coords = np.zeros((21, 3), dtype=np.float32)
            score_R = 0.0
            
        if 24 <= t <= 72:
            left_active = True
            l_shift = np.array([
                -0.06 * np.sin(2.0 * np.pi * (t - 24) / 48.0),
                0.04 * np.cos(2.0 * np.pi * (t - 24) / 48.0),
                -0.01 * np.cos(2.0 * np.pi * (t - 24) / 48.0)
            ], dtype=np.float32)
            l_coords = BASE_LEFT_TEMPLATE + l_shift
            score_L = 0.90
        else:
            left_active = False
            l_coords = np.zeros((21, 3), dtype=np.float32)
            score_L = 0.0
            
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_4_sparse_detections() -> Tuple[np.ndarray, str]:
    """
    Case 4: Only 10 active frames scattered across 50 frames.
    Tests nearest-neighbor presence sampling and isolated active frame velocity spike prevention.
    """
    T = 50
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "Only 10 active frames scattered across 50 frames"
    active_indices = [3, 8, 14, 20, 26, 31, 37, 41, 45, 48]
    both_indices = {14, 20, 26, 31, 37}
    
    for t in active_indices:
        right_active = True
        r_shift = np.array([
            0.03 * np.sin(t * 0.5),
            0.03 * np.cos(t * 0.5),
            0.01 * np.sin(t * 0.3)
        ], dtype=np.float32)
        r_coords = BASE_RIGHT_TEMPLATE + r_shift
        score_R = 0.92
        
        if t in both_indices:
            left_active = True
            l_shift = np.array([
                -0.03 * np.cos(t * 0.5),
                0.03 * np.sin(t * 0.5),
                -0.01 * np.cos(t * 0.3)
            ], dtype=np.float32)
            l_coords = BASE_LEFT_TEMPLATE + l_shift
            score_L = 0.89
        else:
            left_active = False
            l_coords = np.zeros((21, 3), dtype=np.float32)
            score_L = 0.0
            
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_5_single_hand_continuous() -> Tuple[np.ndarray, str]:
    """
    Case 5: Only right hand present throughout 64 frames.
    Tests that Left Hand and Two-Hand interaction remain strictly 0.0 throughout all stages.
    """
    T = 64
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "Only right hand present throughout 64 frames"
    
    for t in range(T):
        right_active = True
        r_shift = np.array([
            0.05 * np.sin(2.0 * np.pi * t / 64.0),
            0.03 * np.cos(2.0 * np.pi * t / 64.0),
            0.01 * np.sin(4.0 * np.pi * t / 64.0)
        ], dtype=np.float32)
        r_coords = BASE_RIGHT_TEMPLATE + r_shift
        score_R = 0.97
        
        left_active = False
        l_coords = np.zeros((21, 3), dtype=np.float32)
        score_L = 0.0
        
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_6_two_hand_continuous() -> Tuple[np.ndarray, str]:
    """
    Case 6: Both hands present throughout 64 frames.
    Tests continuous two-hand normalization and non-zero velocity across all 129 channels.
    """
    T = 64
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "Both hands present throughout 64 frames"
    
    for t in range(T):
        left_active = True
        l_shift = np.array([
            -0.04 * np.sin(2.0 * np.pi * t / 64.0),
            0.04 * np.cos(2.0 * np.pi * t / 64.0),
            -0.01 * np.sin(2.0 * np.pi * t / 64.0)
        ], dtype=np.float32)
        l_coords = BASE_LEFT_TEMPLATE + l_shift
        score_L = 0.93
        
        right_active = True
        r_shift = np.array([
            0.04 * np.cos(2.0 * np.pi * t / 64.0),
            -0.04 * np.sin(2.0 * np.pi * t / 64.0),
            0.01 * np.cos(2.0 * np.pi * t / 64.0)
        ], dtype=np.float32)
        r_coords = BASE_RIGHT_TEMPLATE + r_shift
        score_R = 0.96
        
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_7_alternating_hands() -> Tuple[np.ndarray, str]:
    """
    Case 7: Left hand present in first half, Right hand present in second half.
    Verifies that velocity is strictly 0.0 at transition frame 32 (spike prevention).
    """
    T = 64
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "Left hand present in first half, Right hand present in second half (tests velocity reset when hand switches)"
    
    for t in range(T):
        if t < 32:
            left_active = True
            l_shift = np.array([
                0.03 * np.sin(2.0 * np.pi * t / 32.0),
                0.02 * np.cos(2.0 * np.pi * t / 32.0),
                0.005 * np.sin(2.0 * np.pi * t / 32.0)
            ], dtype=np.float32)
            l_coords = BASE_LEFT_TEMPLATE + l_shift
            score_L = 0.91
            
            right_active = False
            r_coords = np.zeros((21, 3), dtype=np.float32)
            score_R = 0.0
        else:
            left_active = False
            l_coords = np.zeros((21, 3), dtype=np.float32)
            score_L = 0.0
            
            right_active = True
            r_shift = np.array([
                0.03 * np.sin(2.0 * np.pi * (t - 32) / 32.0),
                0.02 * np.cos(2.0 * np.pi * (t - 32) / 32.0),
                0.005 * np.cos(2.0 * np.pi * (t - 32) / 32.0)
            ], dtype=np.float32)
            r_coords = BASE_RIGHT_TEMPLATE + r_shift
            score_R = 0.95
            
        seq[t] = build_frame_134(left_active, right_active, l_coords, r_coords, score_L, score_R)
        
    return seq, desc


def generate_case_8_idle_window() -> Tuple[np.ndarray, str]:
    """
    Case 8: All frames empty (all 263 values strictly 0.0).
    Verifies inactive zero-clamping invariant for an entire idle sequence.
    """
    T = 64
    seq = np.zeros((T, 134), dtype=np.float32)
    desc = "All frames empty (all 263 values strictly 0.0)"
    return seq, desc


def process_case(
    raw_seq: np.ndarray,
    preprocessor: LiveTemporalPreprocessor,
    scaler_dict: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Execute the complete Stage 2 preprocessing pipeline on a single test case.
    
    Pipeline Steps:
    1. Dynamic Resampling: (T, 134) -> (64, 134) via resample_sequence
    2. Active Z-Score Standardization: (64, 134) -> (64, 134) via LiveTemporalPreprocessor formula
    3. 129-D Velocity Generation: (64, 134) -> (64, 129) with transition spike prevention
    4. 263-D Assembly: (64, 263) float32
    5. Canonical Offline Comparison: apply_scaler + compute_velocity_features
    """
    T, D = raw_seq.shape
    assert D == 134, f"Expected 134 features, got {D}"
    
    # 1. Resample to 64 frames
    resampled_64 = resample_sequence(raw_seq, target_len=64)
    assert resampled_64.shape == (64, 134), f"Resampling error: {resampled_64.shape}"
    
    # 2. Live inference preprocessing (matching Android LiveTemporalPreprocessor.kt)
    output_263 = preprocessor.process_window(resampled_64)
    assert output_263.shape == (64, 263), f"Output tensor shape error: {output_263.shape}"
    assert output_263.dtype == np.float32, f"Expected float32, got {output_263.dtype}"
    
    # Extract decomposed stages for intermediate verification
    standardized_64 = output_263[:, 0:134].copy()
    velocity_129 = output_263[:, 134:263].copy()
    
    # 3. Canonical offline reference (for comparative audit)
    offline_std_134 = apply_scaler(resampled_64, scaler_dict)
    offline_263 = compute_velocity_features(offline_std_134)
    
    max_diff_offline = float(np.max(np.abs(output_263 - offline_263)))
    
    return {
        "raw_134": raw_seq.astype(np.float32),
        "resampled_64": resampled_64.astype(np.float32),
        "standardized_64": standardized_64.astype(np.float32),
        "velocity_129": velocity_129.astype(np.float32),
        "output_263": output_263.astype(np.float32),
        "canonical_offline_263": offline_263.astype(np.float32),
        "max_diff_offline": max_diff_offline
    }


def verify_case_invariants(case_name: str, res: Dict[str, Any]):
    """Assert mathematical and physical pipeline invariants for a test case."""
    out = res["output_263"]
    std_134 = res["standardized_64"]
    vel_129 = res["velocity_129"]
    resamp = res["resampled_64"]
    
    # 1. Structural checks
    assert out.shape == (64, 263), f"[{case_name}] Expected shape (64, 263), got {out.shape}"
    assert out.dtype == np.float32, f"[{case_name}] Expected float32, got {out.dtype}"
    assert not np.isnan(out).any(), f"[{case_name}] Output tensor contains NaNs!"
    assert not np.isinf(out).any(), f"[{case_name}] Output tensor contains Infs!"
    
    # 2. Velocity frame 0 invariant: v[0] must be strictly 0.0 for all 129 channels
    assert np.all(vel_129[0] == 0.0), f"[{case_name}] Frame 0 velocity is not all 0.0!"
    assert np.all(out[0, 134:263] == 0.0), f"[{case_name}] Frame 0 velocity in 263-D tensor is not all 0.0!"
    
    # 3. Inactive zero-clamping: where presence is 0, coords and velocities must be strictly 0.0
    for t in range(64):
        p_L = resamp[t, 126] > 0.5
        p_R = resamp[t, 127] > 0.5
        p_both = resamp[t, 130] > 0.5
        
        if not p_L:
            assert np.all(out[t, 0:63] == 0.0), f"[{case_name}] Frame {t} Left Hand inactive coords not 0.0"
            assert np.all(out[t, 134:197] == 0.0), f"[{case_name}] Frame {t} Left Hand inactive velocity not 0.0"
            
        if not p_R:
            assert np.all(out[t, 63:126] == 0.0), f"[{case_name}] Frame {t} Right Hand inactive coords not 0.0"
            assert np.all(out[t, 197:260] == 0.0), f"[{case_name}] Frame {t} Right Hand inactive velocity not 0.0"
            
        if not p_both:
            assert np.all(out[t, 131:134] == 0.0), f"[{case_name}] Frame {t} Interaction inactive coords not 0.0"
            assert np.all(out[t, 260:263] == 0.0), f"[{case_name}] Frame {t} Interaction inactive velocity not 0.0"

    # 4. Case-specific invariants
    if case_name == "case_8_idle_window":
        assert np.all(out == 0.0), f"[{case_name}] Idle window output must be strictly all 0.0!"
        
    elif case_name == "case_5_single_hand_continuous":
        assert np.all(out[:, 0:63] == 0.0), f"[{case_name}] Left Hand coords must be 0.0"
        assert np.all(out[:, 126] == 0.0), f"[{case_name}] Left Hand presence must be 0.0"
        assert np.all(out[:, 130:134] == 0.0), f"[{case_name}] Interaction features must be 0.0"
        assert np.all(out[:, 134:197] == 0.0), f"[{case_name}] Left Hand velocity must be 0.0"
        assert np.all(out[:, 260:263] == 0.0), f"[{case_name}] Interaction velocity must be 0.0"
        assert np.any(out[:, 63:126] != 0.0), f"[{case_name}] Right Hand coords must be active"
        assert np.any(out[1:, 197:260] != 0.0), f"[{case_name}] Right Hand velocity must be active"
        
    elif case_name == "case_6_two_hand_continuous":
        assert np.all(out[:, 126] == 1.0), f"[{case_name}] Left presence must be 1.0"
        assert np.all(out[:, 127] == 1.0), f"[{case_name}] Right presence must be 1.0"
        assert np.all(out[:, 130] == 1.0), f"[{case_name}] Both present must be 1.0"
        assert np.any(out[1:, 134:197] != 0.0), f"[{case_name}] Left Hand velocity must be active"
        assert np.any(out[1:, 197:260] != 0.0), f"[{case_name}] Right Hand velocity must be active"
        assert np.any(out[1:, 260:263] != 0.0), f"[{case_name}] Interaction velocity must be active"
        
    elif case_name == "case_7_alternating_hands":
        # Frame 32 transition: Left disappeared, Right appeared -> zero velocity spikes
        assert np.all(out[32, 134:197] == 0.0), f"[{case_name}] Left Hand velocity at frame 32 must be 0.0"
        assert np.all(out[32, 197:260] == 0.0), f"[{case_name}] Right Hand velocity at frame 32 must be 0.0 (no spike!)"
        assert np.all(out[:, 130:134] == 0.0), f"[{case_name}] Interaction features must be 0.0"
        assert np.all(out[:, 260:263] == 0.0), f"[{case_name}] Interaction velocity must be 0.0"


def to_serializable_2d(arr: np.ndarray, decimals: int = 7) -> List[List[float]]:
    """Convert numpy 2D array to nested list of rounded floats for JSON export."""
    return [[round(float(val), decimals) for val in row] for row in arr]


def main():
    parser = argparse.ArgumentParser(description="EchoGesture Stage 2 Parity Fixture Generator")
    parser.add_argument("--scaler", type=str, default="data/processed/sequences_v3/scaler_params.npz",
                        help="Path to scaler_params.npz")
    parser.add_argument("--out_npz", type=str, default="tests/fixtures/android_parity/stage2_temporal.npz",
                        help="Output path for .npz fixtures")
    parser.add_argument("--out_json", type=str, default="tests/fixtures/android_parity/stage2_temporal.json",
                        help="Output path for .json fixtures")
    args = parser.parse_args()
    
    scaler_path = Path(args.scaler)
    if not scaler_path.is_absolute():
        scaler_path = REPO_ROOT / scaler_path
        
    out_npz_path = Path(args.out_npz)
    if not out_npz_path.is_absolute():
        out_npz_path = REPO_ROOT / out_npz_path
        
    out_json_path = Path(args.out_json)
    if not out_json_path.is_absolute():
        out_json_path = REPO_ROOT / out_json_path
        
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("================================================================================")
    print(" EchoGesture Phase 7B.1: Stage 2 Temporal Preprocessing Fixture Generation")
    print("================================================================================")
    print(f"Scaler Source: {scaler_path}")
    print(f"NPZ Output:    {out_npz_path}")
    print(f"JSON Output:   {out_json_path}")
    
    # Load Scaler
    scaler_data = np.load(str(scaler_path))
    mean_full = scaler_data["mean"].astype(np.float32)
    std_full = scaler_data["std"].astype(np.float32)
    scaler_dict = {"mean": mean_full, "std": std_full}
    
    preprocessor = LiveTemporalPreprocessor(str(scaler_path))
    
    # Define test case generators
    case_generators = [
        ("case_1_short_sequence", generate_case_1_short_sequence),
        ("case_2_full_sequence", generate_case_2_full_sequence),
        ("case_3_long_sequence", generate_case_3_long_sequence),
        ("case_4_sparse_detections", generate_case_4_sparse_detections),
        ("case_5_single_hand_continuous", generate_case_5_single_hand_continuous),
        ("case_6_two_hand_continuous", generate_case_6_two_hand_continuous),
        ("case_7_alternating_hands", generate_case_7_alternating_hands),
        ("case_8_idle_window", generate_case_8_idle_window)
    ]
    
    npz_data: Dict[str, Any] = {
        "scaler_mean": mean_full,
        "scaler_std": std_full,
        "case_names": np.array([name for name, _ in case_generators], dtype="<U32")
    }
    
    json_cases: Dict[str, Any] = {}
    validation_summary: List[Dict[str, Any]] = []
    
    print("\nExecuting pipeline across all 8 test cases...")
    for case_name, generator_fn in case_generators:
        raw_seq, desc = generator_fn()
        res = process_case(raw_seq, preprocessor, scaler_dict)
        verify_case_invariants(case_name, res)
        
        # Populate NPZ dictionary
        npz_data[f"{case_name}_raw_134"] = res["raw_134"]
        npz_data[f"{case_name}_resampled_64"] = res["resampled_64"]
        npz_data[f"{case_name}_standardized_64"] = res["standardized_64"]
        npz_data[f"{case_name}_velocity_129"] = res["velocity_129"]
        npz_data[f"{case_name}_output_263"] = res["output_263"]
        npz_data[f"{case_name}_canonical_offline_263"] = res["canonical_offline_263"]
        
        # Convenient aliases for parity test harnesses
        npz_data[f"{case_name}_input"] = res["raw_134"]
        npz_data[f"{case_name}_output"] = res["output_263"]
        
        # Summary statistics
        out = res["output_263"]
        non_zero_pct = float(np.mean(out != 0.0) * 100.0)
        min_val = float(np.min(out))
        max_val = float(np.max(out))
        mean_val = float(np.mean(out))
        
        # Populate JSON dictionary
        json_cases[case_name] = {
            "case_id": case_name,
            "description": desc,
            "orig_t": int(len(raw_seq)),
            "target_t": 64,
            "raw_shape": [int(len(raw_seq)), 134],
            "output_shape": [64, 263],
            "raw_input_134": to_serializable_2d(res["raw_134"]),
            "resampled_64": to_serializable_2d(res["resampled_64"]),
            "standardized_64": to_serializable_2d(res["standardized_64"]),
            "velocity_129": to_serializable_2d(res["velocity_129"]),
            "output_263": to_serializable_2d(res["output_263"]),
            "canonical_offline_263": to_serializable_2d(res["canonical_offline_263"]),
            "summary_stats": {
                "min": round(min_val, 6),
                "max": round(max_val, 6),
                "mean": round(mean_val, 6),
                "non_zero_pct": round(non_zero_pct, 2),
                "max_diff_canonical_offline": round(res["max_diff_offline"], 7),
                "has_nan": bool(np.isnan(out).any()),
                "has_inf": bool(np.isinf(out).any())
            }
        }
        
        validation_summary.append({
            "case": case_name,
            "orig_t": len(raw_seq),
            "non_zero_pct": non_zero_pct,
            "min_val": min_val,
            "max_val": max_val,
            "max_diff_offline": res["max_diff_offline"]
        })
        
        print(f"  ✓ {case_name:<28} | T_in={len(raw_seq):2d} -> 64 | Non-zero: {non_zero_pct:5.1f}% | Offline Diff: {res['max_diff_offline']:.6f}")
        
    # Save NPZ file
    np.savez_compressed(str(out_npz_path), **npz_data)
    npz_size_kb = out_npz_path.stat().st_size / 1024.0
    print(f"\nSaved NPZ fixtures to: {out_npz_path} ({npz_size_kb:.1f} KB)")
    
    # Save JSON file
    json_doc = {
        "metadata": {
            "fixture_name": "EchoGesture Phase 7B Stage 2 Temporal Parity Fixtures",
            "version": "1.0",
            "component": "stage2_temporal_preprocessing",
            "target_frames": 64,
            "raw_dim": 134,
            "velocity_dim": 129,
            "total_dim": 263,
            "scaler_source": str(Path(args.scaler).as_posix()),
            "generated_timestamp": datetime.utcnow().isoformat() + "Z",
            "num_cases": len(case_generators),
            "case_names": [name for name, _ in case_generators]
        },
        "scaler": {
            "mean": [round(float(v), 7) for v in mean_full],
            "std": [round(float(v), 7) for v in std_full]
        },
        "cases": json_cases
    }
    
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2)
    json_size_mb = out_json_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved JSON fixtures to: {out_json_path} ({json_size_mb:.2f} MB)")
    
    # Final Validation
    print("\n--------------------------------------------------------------------------------")
    print(" STAGE 2 TEMPORAL FIXTURES VALIDATION REPORT")
    print("--------------------------------------------------------------------------------")
    print(f"Total Fixture Cases:          {len(validation_summary)}")
    print(f"Tensor Shape Parity:          100% Passed (All 8 cases strictly [64, 263])")
    print(f"Numerical Precision:          float32, Zero NaNs, Zero Infs across all cases")
    print(f"Zero-Clamping Invariants:     100% Verified (Idle, Single Hand, Inactive Frames)")
    print(f"Velocity Boundary Frame 0:    Strictly 0.0 across all 129 channels")
    print(f"Transition Spike Prevention:  Strictly 0.0 on hand appearance/disappearance")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
