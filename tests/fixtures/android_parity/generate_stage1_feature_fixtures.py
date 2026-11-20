"""
EchoGesture Phase 7B.1: Android Parity Stage 1 Feature Fixtures Generator
Generates deterministic reference fixtures from the canonical Python 134-D
feature extraction pipeline (LiveLandmarkExtractor in scripts/live_inference.py).

Vector Layout (134-D):
  [0:63]   : Slot 0 (Viewer-Left Hand) 21 normalized landmarks (x, y, z)
  [63:126] : Slot 1 (Viewer-Right Hand) 21 normalized landmarks (x, y, z)
  [126]    : Left Hand Presence flag (1.0 or 0.0)
  [127]    : Right Hand Presence flag (1.0 or 0.0)
  [128]    : Left Hand handedness classification score ([0.0, 1.0])
  [129]    : Right Hand handedness classification score ([0.0, 1.0])
  [130]    : Both Hands Co-presence flag (1.0 or 0.0)
  [131]    : Aspect-ratio corrected inter-hand wrist Euclidean distance normalized by frame diagonal
  [132:134]: Aspect-ratio corrected inter-hand wrist offset (dx, dy) normalized by frame diagonal

Outputs:
  - tests/fixtures/android_parity/stage1_features.npz
  - tests/fixtures/android_parity/stage1_features.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_inference import LiveLandmarkExtractor

# Canonical hand offset pattern relative to wrist (21 landmarks)
# Joint indices:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)
CANONICAL_HAND_OFFSETS = np.array([
    [ 0.00,  0.00,  0.00],   # 0: Wrist
    [-0.03, -0.02, -0.01],  # 1: Thumb CMC
    [-0.05, -0.05, -0.02],  # 2: Thumb MCP
    [-0.07, -0.07, -0.02],  # 3: Thumb IP
    [-0.08, -0.09, -0.03],  # 4: Thumb Tip
    [-0.03, -0.08, -0.01],  # 5: Index MCP
    [-0.04, -0.12, -0.02],  # 6: Index PIP
    [-0.04, -0.15, -0.02],  # 7: Index DIP
    [-0.04, -0.18, -0.03],  # 8: Index Tip
    [-0.01, -0.09,  0.00],  # 9: Middle MCP
    [-0.01, -0.13, -0.01],  # 10: Middle PIP
    [-0.01, -0.16, -0.02],  # 11: Middle DIP
    [-0.01, -0.19, -0.02],  # 12: Middle Tip
    [ 0.01, -0.08, -0.01],  # 13: Ring MCP
    [ 0.02, -0.12, -0.02],  # 14: Ring PIP
    [ 0.02, -0.15, -0.02],  # 15: Ring DIP
    [ 0.02, -0.18, -0.02],  # 16: Ring Tip
    [ 0.03, -0.07, -0.01],  # 17: Pinky MCP
    [ 0.05, -0.10, -0.02],  # 18: Pinky PIP
    [ 0.06, -0.13, -0.02],  # 19: Pinky DIP
    [ 0.07, -0.15, -0.03],  # 20: Pinky Tip
], dtype=np.float32)


def generate_hand_landmarks(wrist_xyz: Tuple[float, float, float], scale: float = 1.0) -> np.ndarray:
    """Generate 21 (x, y, z) landmarks anchored at wrist_xyz with canonical anatomical structure."""
    wrist = np.array(wrist_xyz, dtype=np.float32)
    landmarks = wrist + CANONICAL_HAND_OFFSETS * scale
    return landmarks.astype(np.float32)


# Mock MediaPipe data structures for deterministic testing
class MockLandmark:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class MockHandLandmarks:
    def __init__(self, landmarks_21: np.ndarray):
        self.landmark = [MockLandmark(lm[0], lm[1], lm[2]) for lm in landmarks_21]


class MockClassification:
    def __init__(self, label: str, score: float):
        self.label = str(label)
        self.score = float(score)


class MockHandedness:
    def __init__(self, label: str, score: float):
        self.classification = [MockClassification(label, score)]


class MockMediaPipeResults:
    def __init__(
        self,
        hand_landmarks: Optional[List[MockHandLandmarks]] = None,
        handedness: Optional[List[MockHandedness]] = None
    ):
        self.multi_hand_landmarks = hand_landmarks
        self.multi_handedness = handedness


def build_mock_detection(
    hands: List[Dict[str, Any]]
) -> MockMediaPipeResults:
    """
    Constructs a MockMediaPipeResults object from a list of hand specifications.
    Each hand dict should contain:
      - 'landmarks': (21, 3) np.ndarray
      - 'label': 'Left' or 'Right'
      - 'score': float confidence in [0.0, 1.0]
    """
    if not hands:
        return MockMediaPipeResults(None, None)

    lms_list = []
    handedness_list = []
    for h in hands:
        lms_list.append(MockHandLandmarks(h["landmarks"]))
        handedness_list.append(MockHandedness(h["label"], h["score"]))

    return MockMediaPipeResults(lms_list, handedness_list)


def process_deterministic_frame(
    extractor: LiveLandmarkExtractor,
    hands: Optional[List[Dict[str, Any]]],
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Feed a deterministic hand configuration through the canonical LiveLandmarkExtractor.
    If hands is None, executes real MediaPipe on a blank black frame.
    """
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)

    if hands is None:
        # Use live MediaPipe detector on blank frame
        flat_134, telemetry, _ = extractor.process_frame(dummy_frame)
    else:
        # Mock detector.process for deterministic landmark input
        mock_results = build_mock_detection(hands)
        original_process = extractor.detector.process
        extractor.detector.process = lambda rgb: mock_results
        try:
            flat_134, telemetry, _ = extractor.process_frame(dummy_frame)
        finally:
            extractor.detector.process = original_process

    return flat_134, telemetry


def serialize_frame_breakdown(flat_134: np.ndarray, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Create a structured JSON-serializable breakdown of a 134-D feature vector."""
    return {
        "features_134": [float(x) for x in flat_134],
        "telemetry": telemetry,
        "breakdown": {
            "slot0_left_landmarks": [[float(c) for c in lm] for lm in flat_134[0:63].reshape(21, 3)],
            "slot1_right_landmarks": [[float(c) for c in lm] for lm in flat_134[63:126].reshape(21, 3)],
            "slot0_left_wrist": [float(x) for x in flat_134[0:3]],
            "slot1_right_wrist": [float(x) for x in flat_134[63:66]],
            "left_presence": float(flat_134[126]),
            "right_presence": float(flat_134[127]),
            "left_score": float(flat_134[128]),
            "right_score": float(flat_134[129]),
            "both_present": float(flat_134[130]),
            "inter_hand_dist_geom": float(flat_134[131]),
            "inter_hand_offset_geom": [float(flat_134[132]), float(flat_134[133])]
        }
    }


def generate_all_fixtures() -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate all 7 test cases specified in EchoGesture Phase 7B.1 contract.
    """
    fixtures_npz: Dict[str, np.ndarray] = {}
    fixtures_json: Dict[str, Any] = {
        "metadata": {
            "project": "EchoGesture",
            "phase": "Phase 7B.1",
            "title": "Stage 1 Landmark Feature Parity Fixtures",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_reference": "scripts/live_inference.py:LiveLandmarkExtractor",
            "contract_document": "reports/production_inference_contract.md",
            "feature_dim": 134,
            "hand_slots": {
                "slot_0": "Viewer-Left Hand (MediaPipe 'Left')",
                "slot_1": "Viewer-Right Hand (MediaPipe 'Right')"
            },
            "layout_slices": {
                "slot0_left_landmarks": [0, 63],
                "slot1_right_landmarks": [63, 126],
                "left_presence": 126,
                "right_presence": 127,
                "left_score": 128,
                "right_score": 129,
                "both_present": 130,
                "inter_hand_dist_geom": 131,
                "inter_hand_offset_dx": 132,
                "inter_hand_offset_dy": 133
            }
        },
        "cases": {}
    }

    extractor = LiveLandmarkExtractor()

    try:
        # =====================================================================
        # Case 1: case_1_left_only
        # Left hand present, Right absent.
        # =====================================================================
        print("Generating Case 1: left_only...")
        extractor.reset_tracking()
        left_lms_c1 = generate_hand_landmarks(wrist_xyz=(0.30, 0.60, 0.00), scale=1.0)
        c1_hands = [
            {"label": "Left", "score": 0.95, "landmarks": left_lms_c1}
        ]
        feat_c1, telem_c1 = process_deterministic_frame(extractor, c1_hands, width=640, height=480)
        fixtures_npz["case_1_left_only"] = feat_c1
        fixtures_json["cases"]["case_1_left_only"] = {
            "description": "Left hand present (p=0.95), Right hand absent. Right slot zero-masked.",
            "frame_shape": [480, 640],
            **serialize_frame_breakdown(feat_c1, telem_c1)
        }

        # =====================================================================
        # Case 2: case_2_right_only
        # Right hand present, Left absent.
        # =====================================================================
        print("Generating Case 2: right_only...")
        extractor.reset_tracking()
        right_lms_c2 = generate_hand_landmarks(wrist_xyz=(0.70, 0.60, 0.00), scale=1.0)
        c2_hands = [
            {"label": "Right", "score": 0.92, "landmarks": right_lms_c2}
        ]
        feat_c2, telem_c2 = process_deterministic_frame(extractor, c2_hands, width=640, height=480)
        fixtures_npz["case_2_right_only"] = feat_c2
        fixtures_json["cases"]["case_2_right_only"] = {
            "description": "Right hand present (p=0.92), Left hand absent. Left slot zero-masked.",
            "frame_shape": [480, 640],
            **serialize_frame_breakdown(feat_c2, telem_c2)
        }

        # =====================================================================
        # Case 3: case_3_both_hands
        # Both hands present, standard spatial separation.
        # =====================================================================
        print("Generating Case 3: both_hands...")
        extractor.reset_tracking()
        left_lms_c3 = generate_hand_landmarks(wrist_xyz=(0.30, 0.60, 0.00), scale=1.0)
        right_lms_c3 = generate_hand_landmarks(wrist_xyz=(0.70, 0.50, 0.00), scale=1.0)
        c3_hands = [
            {"label": "Left", "score": 0.98, "landmarks": left_lms_c3},
            {"label": "Right", "score": 0.94, "landmarks": right_lms_c3}
        ]
        feat_c3, telem_c3 = process_deterministic_frame(extractor, c3_hands, width=640, height=480)
        fixtures_npz["case_3_both_hands"] = feat_c3
        fixtures_json["cases"]["case_3_both_hands"] = {
            "description": "Both hands present: Left wrist=(0.3, 0.6), Right wrist=(0.7, 0.5) in 640x480 frame.",
            "frame_shape": [480, 640],
            **serialize_frame_breakdown(feat_c3, telem_c3)
        }

        # =====================================================================
        # Case 4: case_4_missing_hands
        # Neither hand detected (strictly zero-masked).
        # =====================================================================
        print("Generating Case 4: missing_hands...")
        extractor.reset_tracking()
        # Test with real blank image through MediaPipe
        feat_c4, telem_c4 = process_deterministic_frame(extractor, hands=None, width=640, height=480)
        fixtures_npz["case_4_missing_hands"] = feat_c4
        fixtures_json["cases"]["case_4_missing_hands"] = {
            "description": "Neither hand detected. All 134 features strictly zero-masked.",
            "frame_shape": [480, 640],
            **serialize_frame_breakdown(feat_c4, telem_c4)
        }

        # =====================================================================
        # Case 5: case_5_transition_sequence
        # 5-frame sequence: appearance, movement, and disappearance.
        # =====================================================================
        print("Generating Case 5: transition_sequence (5 frames)...")
        extractor.reset_tracking()
        c5_frames_feat: List[np.ndarray] = []
        c5_frames_json: List[Dict[str, Any]] = []

        # Define 5 distinct states:
        # F0: No hands
        # F1: Left appears at (0.35, 0.65)
        # F2: Left moves to (0.33, 0.60), Right appears at (0.67, 0.55)
        # F3: Left moves to (0.30, 0.55), Right moves to (0.70, 0.50)
        # F4: Left disappears, Right remains at (0.72, 0.48)
        seq_defs = [
            # Frame 0: No hands
            {"name": "Frame 0: Empty frame", "hands": []},
            # Frame 1: Left appears
            {"name": "Frame 1: Left hand appears", "hands": [
                {"label": "Left", "score": 0.90, "landmarks": generate_hand_landmarks((0.35, 0.65, 0.00))}
            ]},
            # Frame 2: Both present
            {"name": "Frame 2: Right hand appears, Left moves", "hands": [
                {"label": "Left", "score": 0.93, "landmarks": generate_hand_landmarks((0.33, 0.60, 0.00))},
                {"label": "Right", "score": 0.91, "landmarks": generate_hand_landmarks((0.67, 0.55, 0.00))}
            ]},
            # Frame 3: Both move
            {"name": "Frame 3: Both hands move co-present", "hands": [
                {"label": "Left", "score": 0.95, "landmarks": generate_hand_landmarks((0.30, 0.55, 0.00))},
                {"label": "Right", "score": 0.94, "landmarks": generate_hand_landmarks((0.70, 0.50, 0.00))}
            ]},
            # Frame 4: Left disappears, Right stays
            {"name": "Frame 4: Left disappears, Right remains", "hands": [
                {"label": "Right", "score": 0.96, "landmarks": generate_hand_landmarks((0.72, 0.48, 0.00))}
            ]}
        ]

        for f_idx, sdef in enumerate(seq_defs):
            feat_f, telem_f = process_deterministic_frame(extractor, sdef["hands"], width=640, height=480)
            c5_frames_feat.append(feat_f)
            c5_frames_json.append({
                "frame_index": f_idx,
                "state_description": sdef["name"],
                **serialize_frame_breakdown(feat_f, telem_f)
            })

        c5_seq_array = np.stack(c5_frames_feat, axis=0)  # Shape: (5, 134)
        fixtures_npz["case_5_transition_sequence"] = c5_seq_array
        fixtures_json["cases"]["case_5_transition_sequence"] = {
            "description": "5-frame sequence verifying hand appearance, movement, and disappearance state transitions.",
            "sequence_length": 5,
            "frame_shape": [480, 640],
            "frames": c5_frames_json
        }

        # =====================================================================
        # Case 6: case_6_hungarian_tracking
        # Ambiguous duplicate labels resolved via Hungarian temporal tracking.
        # =====================================================================
        print("Generating Case 6: hungarian_tracking...")
        extractor.reset_tracking()

        # Frame 0: Clean initialization with Left at 0.25 and Right at 0.75
        f0_hands = [
            {"label": "Left", "score": 0.95, "landmarks": generate_hand_landmarks((0.25, 0.50, 0.00))},
            {"label": "Right", "score": 0.95, "landmarks": generate_hand_landmarks((0.75, 0.50, 0.00))}
        ]
        feat_h0, telem_h0 = process_deterministic_frame(extractor, f0_hands, width=640, height=480)

        # Frame 1: MediaPipe outputs duplicate label 'Left' for BOTH hands!
        # Hand 0 in list is near prev_wrist[1] (Right: 0.74, 0.51)
        # Hand 1 in list is near prev_wrist[0] (Left: 0.26, 0.49)
        # Hungarian nearest-wrist assignment MUST assign Hand 1 -> Slot 0 and Hand 0 -> Slot 1!
        f1_duplicate_hands = [
            {"label": "Left", "score": 0.88, "landmarks": generate_hand_landmarks((0.74, 0.51, 0.00))},
            {"label": "Left", "score": 0.91, "landmarks": generate_hand_landmarks((0.26, 0.49, 0.00))}
        ]
        feat_h1, telem_h1 = process_deterministic_frame(extractor, f1_duplicate_hands, width=640, height=480)

        c6_seq_array = np.stack([feat_h0, feat_h1], axis=0)  # Shape: (2, 134)
        fixtures_npz["case_6_hungarian_tracking"] = c6_seq_array
        fixtures_npz["case_6_hungarian_frame0"] = feat_h0
        fixtures_npz["case_6_hungarian_frame1"] = feat_h1

        fixtures_json["cases"]["case_6_hungarian_tracking"] = {
            "description": "Frame 0 initializes slots (L=(0.25,0.5), R=(0.75,0.5)). "
                           "Frame 1 produces duplicate 'Left' labels for both hands; "
                           "Hungarian nearest-wrist tracking swaps candidate order to maintain slot continuity.",
            "sequence_length": 2,
            "frame_shape": [480, 640],
            "frames": [
                {
                    "frame_index": 0,
                    "state_description": "Initial frame with distinct Left and Right hands",
                    **serialize_frame_breakdown(feat_h0, telem_h0)
                },
                {
                    "frame_index": 1,
                    "state_description": "Duplicate 'Left' labels; Hungarian tracking assigns Hand 1->Slot 0, Hand 0->Slot 1",
                    **serialize_frame_breakdown(feat_h1, telem_h1)
                }
            ]
        }

        # =====================================================================
        # Case 7: case_7_aspect_ratios
        # Testing aspect-ratio normalization invariance across 640x480, 1280x720 (and 1920x1080).
        # =====================================================================
        print("Generating Case 7: aspect_ratios...")
        # Identical normalized coordinates: Left wrist (0.30, 0.50), Right wrist (0.70, 0.50)
        c7_hands = [
            {"label": "Left", "score": 0.95, "landmarks": generate_hand_landmarks((0.30, 0.50, 0.00))},
            {"label": "Right", "score": 0.95, "landmarks": generate_hand_landmarks((0.70, 0.50, 0.00))}
        ]

        # 1. 640x480 (4:3 aspect ratio, diag = 800)
        extractor.reset_tracking()
        feat_ar_640, telem_ar_640 = process_deterministic_frame(extractor, c7_hands, width=640, height=480)

        # 2. 1280x720 (16:9 aspect ratio, diag = sqrt(1280^2 + 720^2) ≈ 1468.605)
        extractor.reset_tracking()
        feat_ar_1280, telem_ar_1280 = process_deterministic_frame(extractor, c7_hands, width=1280, height=720)

        # 3. 1920x1080 (16:9 aspect ratio, diag = sqrt(1920^2 + 1080^2) ≈ 2202.907)
        # Note: 1920x1080 has the exact same aspect ratio as 1280x720, so normalized geometric features must match!
        extractor.reset_tracking()
        feat_ar_1920, telem_ar_1920 = process_deterministic_frame(extractor, c7_hands, width=1920, height=1080)

        fixtures_npz["case_7_aspect_ratio_640x480"] = feat_ar_640
        fixtures_npz["case_7_aspect_ratio_1280x720"] = feat_ar_1280
        fixtures_npz["case_7_aspect_ratio_1920x1080"] = feat_ar_1920

        fixtures_json["cases"]["case_7_aspect_ratios"] = {
            "description": "Aspect-ratio normalization across 640x480 (4:3), 1280x720 (16:9), and 1920x1080 (16:9). "
                           "Demonstrates aspect ratio scaling while preserving resolution invariance within identical aspect ratios.",
            "resolutions": {
                "640x480": {
                    "width": 640,
                    "height": 480,
                    "aspect_ratio": "4:3",
                    "diagonal_px": float(np.sqrt(640**2 + 480**2)),
                    **serialize_frame_breakdown(feat_ar_640, telem_ar_640)
                },
                "1280x720": {
                    "width": 1280,
                    "height": 720,
                    "aspect_ratio": "16:9",
                    "diagonal_px": float(np.sqrt(1280**2 + 720**2)),
                    **serialize_frame_breakdown(feat_ar_1280, telem_ar_1280)
                },
                "1920x1080": {
                    "width": 1920,
                    "height": 1080,
                    "aspect_ratio": "16:9",
                    "diagonal_px": float(np.sqrt(1920**2 + 1080**2)),
                    **serialize_frame_breakdown(feat_ar_1920, telem_ar_1920)
                }
            }
        }

    finally:
        extractor.close()

    return fixtures_npz, fixtures_json


def validate_fixtures(npz_dict: Dict[str, np.ndarray], json_dict: Dict[str, Any]) -> None:
    """
    Strict validation of all generated fixtures:
      1. Dimension checks: each vector must be strictly (134,) float32
      2. No NaN or Inf values anywhere
      3. Zero-masking exactness on inactive slots
      4. Mathematical correctness of aspect-ratio geometry
      5. Hungarian tracking slot continuity verification
    """
    print("\n--- Validating Generated Fixtures ---")

    for key, arr in npz_dict.items():
        # Check dtype
        assert arr.dtype == np.float32, f"Array {key} dtype {arr.dtype} is not float32!"
        # Check for NaNs or Infs
        assert not np.isnan(arr).any(), f"Array {key} contains NaN values!"
        assert not np.isinf(arr).any(), f"Array {key} contains Inf values!"

        # Dimension checks
        if arr.ndim == 1:
            assert arr.shape == (134,), f"Array {key} shape {arr.shape} is not (134,)!"
        elif arr.ndim == 2:
            assert arr.shape[1] == 134, f"Sequence {key} feature dimension {arr.shape[1]} is not 134!"
        print(f"  [PASS] Array {key:30s} shape={str(arr.shape):10s} dtype={arr.dtype} (no NaNs/Infs)")

    # 1. Validate Case 1: left_only
    c1 = npz_dict["case_1_left_only"]
    assert c1[126] == 1.0, "Case 1: left presence must be 1.0"
    assert c1[127] == 0.0, "Case 1: right presence must be 0.0"
    assert abs(c1[128] - 0.95) < 1e-5, f"Case 1: left score must be 0.95, got {c1[128]}"
    assert c1[129] == 0.0, "Case 1: right score must be 0.0"
    assert c1[130] == 0.0, "Case 1: both present must be 0.0"
    assert c1[131] == 0.0, "Case 1: inter-hand distance must be 0.0"
    assert np.all(c1[63:126] == 0.0), "Case 1: right landmarks must be strictly zero-masked"
    assert np.all(c1[132:134] == 0.0), "Case 1: inter-hand offset must be strictly zero-masked"
    print("  [PASS] Case 1 (left_only) zero-masking and presence flags verified.")

    # 2. Validate Case 2: right_only
    c2 = npz_dict["case_2_right_only"]
    assert c2[126] == 0.0, "Case 2: left presence must be 0.0"
    assert c2[127] == 1.0, "Case 2: right presence must be 1.0"
    assert c2[128] == 0.0, "Case 2: left score must be 0.0"
    assert abs(c2[129] - 0.92) < 1e-5, f"Case 2: right score must be 0.92, got {c2[129]}"
    assert c2[130] == 0.0, "Case 2: both present must be 0.0"
    assert c2[131] == 0.0, "Case 2: inter-hand distance must be 0.0"
    assert np.all(c2[0:63] == 0.0), "Case 2: left landmarks must be strictly zero-masked"
    assert np.all(c2[132:134] == 0.0), "Case 2: inter-hand offset must be strictly zero-masked"
    print("  [PASS] Case 2 (right_only) zero-masking and presence flags verified.")

    # 3. Validate Case 3: both_hands
    c3 = npz_dict["case_3_both_hands"]
    assert c3[126] == 1.0 and c3[127] == 1.0 and c3[130] == 1.0, "Case 3: presence flags must all be 1.0"
    assert abs(c3[128] - 0.98) < 1e-5, f"Case 3: left score expected 0.98, got {c3[128]}"
    assert abs(c3[129] - 0.94) < 1e-5, f"Case 3: right score expected 0.94, got {c3[129]}"
    # Expected geometry in 640x480:
    # L_wrist=(0.3, 0.6), R_wrist=(0.7, 0.5)
    # dx_px = 0.4 * 640 = 256, dy_px = -0.1 * 480 = -48
    # diag = 800
    # inter_dist = sqrt(256^2 + (-48)^2) / 800 = sqrt(67840) / 800 ≈ 0.3255764
    # offset_dx = 256 / 800 = 0.32, offset_dy = -48 / 800 = -0.06
    expected_dist_c3 = np.sqrt(256.0**2 + (-48.0)**2) / 800.0
    assert abs(c3[131] - expected_dist_c3) < 1e-5, f"Case 3: inter_dist mismatch: {c3[131]} vs {expected_dist_c3}"
    assert abs(c3[132] - 0.32) < 1e-5, f"Case 3: offset_dx mismatch: {c3[132]} vs 0.32"
    assert abs(c3[133] - (-0.06)) < 1e-5, f"Case 3: offset_dy mismatch: {c3[133]} vs -0.06"
    print("  [PASS] Case 3 (both_hands) geometric invariants verified.")

    # 4. Validate Case 4: missing_hands
    c4 = npz_dict["case_4_missing_hands"]
    assert np.all(c4 == 0.0), "Case 4: all 134 features must be strictly 0.0"
    print("  [PASS] Case 4 (missing_hands) exact zero-masking verified.")

    # 5. Validate Case 5: transition_sequence
    c5 = npz_dict["case_5_transition_sequence"]
    assert c5.shape == (5, 134), f"Case 5: shape expected (5, 134), got {c5.shape}"
    # Frame 0: empty
    assert np.all(c5[0] == 0.0), "Case 5 Frame 0 must be all zeros"
    # Frame 1: left only
    assert c5[1, 126] == 1.0 and c5[1, 127] == 0.0 and c5[1, 130] == 0.0
    assert np.all(c5[1, 63:126] == 0.0)
    # Frame 2: both present
    assert c5[2, 126] == 1.0 and c5[2, 127] == 1.0 and c5[2, 130] == 1.0
    assert c5[2, 131] > 0.0
    # Frame 3: both present
    assert c5[3, 126] == 1.0 and c5[3, 127] == 1.0 and c5[3, 130] == 1.0
    # Frame 4: right only
    assert c5[4, 126] == 0.0 and c5[4, 127] == 1.0 and c5[4, 130] == 0.0
    assert np.all(c5[4, 0:63] == 0.0)
    print("  [PASS] Case 5 (transition_sequence) 5-frame transitions verified.")

    # 6. Validate Case 6: hungarian_tracking
    c6 = npz_dict["case_6_hungarian_tracking"]
    assert c6.shape == (2, 134), f"Case 6: shape expected (2, 134), got {c6.shape}"
    # Frame 0: Slot 0 wrist x ≈ 0.25, Slot 1 wrist x ≈ 0.75
    assert abs(c6[0, 0] - 0.25) < 1e-4, f"Case 6 Frame 0: Slot 0 wrist x expected 0.25, got {c6[0, 0]}"
    assert abs(c6[0, 63] - 0.75) < 1e-4, f"Case 6 Frame 0: Slot 1 wrist x expected 0.75, got {c6[0, 63]}"
    # Frame 1: Despite Candidate 0 being at 0.74 (and first in list),
    # Hungarian tracking must assign Candidate 1 (wrist x ≈ 0.26) to Slot 0,
    # and Candidate 0 (wrist x ≈ 0.74) to Slot 1!
    assert abs(c6[1, 0] - 0.26) < 1e-4, f"Case 6 Frame 1: Slot 0 wrist x expected 0.26, got {c6[1, 0]}"
    assert abs(c6[1, 63] - 0.74) < 1e-4, f"Case 6 Frame 1: Slot 1 wrist x expected 0.74, got {c6[1, 63]}"
    print("  [PASS] Case 6 (hungarian_tracking) slot continuity resolution verified.")

    # 7. Validate Case 7: aspect_ratios
    c7_640 = npz_dict["case_7_aspect_ratio_640x480"]
    c7_1280 = npz_dict["case_7_aspect_ratio_1280x720"]
    c7_1920 = npz_dict["case_7_aspect_ratio_1920x1080"]

    # In 640x480 (4:3):
    # dx_norm = 0.40, dy_norm = 0.0
    # dx_px = 0.4 * 640 = 256, dy_px = 0, diag = 800
    # inter_dist = 256 / 800 = 0.3200
    assert abs(c7_640[131] - 0.3200) < 1e-5, f"Case 7 640x480: expected 0.3200, got {c7_640[131]}"
    assert abs(c7_640[132] - 0.3200) < 1e-5, f"Case 7 640x480: offset_dx expected 0.3200, got {c7_640[132]}"

    # In 1280x720 (16:9):
    # dx_px = 0.4 * 1280 = 512, diag = sqrt(1280^2 + 720^2) = 1468.60478
    # inter_dist = 512 / 1468.60478 ≈ 0.3486302
    expected_dist_1280 = 512.0 / np.sqrt(1280.0**2 + 720.0**2)
    assert abs(c7_1280[131] - expected_dist_1280) < 1e-5, f"Case 7 1280x720: expected {expected_dist_1280}, got {c7_1280[131]}"

    # In 1920x1080 (16:9):
    # Same aspect ratio as 1280x720 -> inter_dist must be IDENTICAL within float precision!
    assert abs(c7_1280[131] - c7_1920[131]) < 1e-6, "Case 7: 1280x720 and 1920x1080 must have identical normalized distance"
    assert abs(c7_1280[132] - c7_1920[132]) < 1e-6, "Case 7: 1280x720 and 1920x1080 must have identical normalized offset dx"
    print("  [PASS] Case 7 (aspect_ratios) diagonal normalization and scale invariance verified.")

    print("\n--- All Fixture Validation Checks Passed Successfully! ---")


def main():
    output_dir = PROJECT_ROOT / "tests" / "fixtures" / "android_parity"
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "stage1_features.npz"
    json_path = output_dir / "stage1_features.json"

    print(f"Generating EchoGesture Phase 7B.1 Reference Fixtures...")
    fixtures_npz, fixtures_json = generate_all_fixtures()

    # Validate before writing
    validate_fixtures(fixtures_npz, fixtures_json)

    # Save .npz
    print(f"\nSaving NPZ fixtures to: {npz_path}")
    np.savez_compressed(npz_path, **fixtures_npz)

    # Save .json
    print(f"Saving JSON fixtures to: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fixtures_json, f, indent=2)

    # Re-read and verify saved files on disk
    print("\nVerifying written artifacts on disk...")
    assert npz_path.exists(), f"File {npz_path} does not exist!"
    assert json_path.exists(), f"File {json_path} does not exist!"

    loaded_npz = np.load(npz_path)
    print(f"NPZ keys ({len(loaded_npz.files)} total): {list(loaded_npz.files)}")
    for k in loaded_npz.files:
        arr = loaded_npz[k]
        assert arr.dtype == np.float32
        assert not np.isnan(arr).any()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    print(f"JSON cases ({len(loaded_json['cases'])} total): {list(loaded_json['cases'].keys())}")

    print(f"\nArtifact size summary:")
    print(f"  {npz_path.name}: {npz_path.stat().st_size:,} bytes")
    print(f"  {json_path.name}: {json_path.stat().st_size:,} bytes")
    print("\nStage 1 reference fixture generation complete.")


if __name__ == "__main__":
    main()
