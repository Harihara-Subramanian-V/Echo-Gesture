"""
EchoGesture Phase 6E: Physical-Camera Evaluation Regression Test Suite
Validates the physical-camera evaluation protocol, geometric and photometric
transformations, temporal modifications, occlusion recovery, idle decay,
and CSV report schema integrity.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_camera_runtime import (
    get_protocol_condition_registry,
    apply_distance_nominal,
    apply_distance_close,
    apply_distance_far,
    apply_position_center,
    apply_position_left_offset,
    apply_position_right_offset,
    apply_position_standing,
    apply_position_seated,
    apply_angle_0deg,
    apply_angle_pitch_down_30deg,
    apply_angle_yaw_30deg,
    apply_lighting_nominal,
    apply_lighting_low_light,
    apply_lighting_harsh_backlight,
    apply_temporal_speed,
    apply_temporal_occlusion,
    create_resting_sequence,
    create_consecutive_sequence,
    create_idle_baseline_sequence,
    CameraRuntimeEvaluator,
    build_reference_clip_pool
)


class TestCameraEvaluationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.scaler_path = str(cls.repo_root / "data" / "processed" / "sequences_v3" / "scaler_params.npz")
        cls.model_path = str(cls.repo_root / "models" / "quantized" / "echogesture_exp_d_int8.tflite")
        
        # Verify prerequisite model and scaler files exist
        assert os.path.exists(cls.scaler_path), f"Missing scaler at {cls.scaler_path}"
        assert os.path.exists(cls.model_path), f"Missing model at {cls.model_path}"
        
        # Create standard synthetic test frame (480x640x3)
        cls.test_frame = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)
        # Add high-contrast circle simulating head/hands
        import cv2
        cv2.circle(cls.test_frame, (320, 240), 60, (220, 200, 180), -1)

    def test_01_protocol_specification_completeness(self):
        """Test that protocol registry contains all 10 required physical condition categories."""
        registry = get_protocol_condition_registry()
        
        # Required categories from specification
        expected_categories = [
            "Distance", "Position", "Angle", "Lighting",
            "HandConfig", "Occlusion", "Speed", "Resting",
            "Consecutive", "FalsePositive"
        ]
        registered_categories = set(spec.category for spec in registry.values())
        
        for cat in expected_categories:
            self.assertIn(cat, registered_categories, f"Missing required category: {cat}")
            
        # Required specific condition names
        required_conditions = [
            "distance_nominal_0.8m", "distance_close_0.5m", "distance_far_1.5m",
            "position_center", "position_left_offset", "position_right_offset",
            "position_standing", "position_seated",
            "angle_0deg", "angle_pitch_down_30deg", "angle_yaw_30deg",
            "lighting_nominal", "lighting_low_light", "lighting_harsh_backlight",
            "hand_config_one_handed", "hand_config_two_handed",
            "occlusion_hand_drop", "occlusion_tracking_blackout", "occlusion_rapid_return",
            "speed_rapid", "speed_standard", "speed_prolonged",
            "signing_to_rest", "consecutive_signing", "false_positive_idle"
        ]
        for c_name in required_conditions:
            self.assertIn(c_name, registry, f"Missing required condition: {c_name}")

    def test_02_distance_geometric_transformations(self):
        """Test distance zoom (close) and downscale (far) geometry, bounds, and dtypes."""
        frame = self.test_frame.copy()
        
        # Nominal: exact copy
        nom = apply_distance_nominal(frame)
        self.assertEqual(nom.shape, (480, 640, 3))
        self.assertEqual(nom.dtype, np.uint8)
        np.testing.assert_array_equal(nom, frame)
        
        # Close (~0.5m, 1.4x zoom): center crop scaled up
        close = apply_distance_close(frame, zoom=1.4)
        self.assertEqual(close.shape, (480, 640, 3))
        self.assertEqual(close.dtype, np.uint8)
        # Center circle should be magnified
        self.assertFalse(np.array_equal(close, frame))
        
        # Far (~1.5m, 0.6x downscale): border canvas padding
        far = apply_distance_far(frame, scale=0.6)
        self.assertEqual(far.shape, (480, 640, 3))
        self.assertEqual(far.dtype, np.uint8)
        # Outer border pixels should match border padding (value 40)
        self.assertEqual(far[5, 5, 0], 40)
        self.assertEqual(far[475, 635, 0], 40)

    def test_03_position_spatial_transformations(self):
        """Test spatial translations (left offset, right offset, standing, seated)."""
        frame = self.test_frame.copy()
        h, w = frame.shape[:2]
        
        # Left offset
        left = apply_position_left_offset(frame, frac=-0.18)
        self.assertEqual(left.shape, (h, w, 3))
        self.assertEqual(left.dtype, np.uint8)
        # Right border should have padding
        self.assertEqual(left[240, 630, 0], 40)
        
        # Right offset
        right = apply_position_right_offset(frame, frac=0.18)
        self.assertEqual(right.shape, (h, w, 3))
        self.assertEqual(right.dtype, np.uint8)
        # Left border should have padding
        self.assertEqual(right[240, 10, 0], 40)
        
        # Standing (upward translation)
        standing = apply_position_standing(frame)
        self.assertEqual(standing.shape, (h, w, 3))
        # Bottom border should have padding
        self.assertEqual(standing[475, 320, 0], 40)
        
        # Seated (downward translation)
        seated = apply_position_seated(frame)
        self.assertEqual(seated.shape, (h, w, 3))
        # Top border should have padding
        self.assertEqual(seated[5, 320, 0], 40)

    def test_04_angle_perspective_transformations(self):
        """Test 3D perspective homographies for 30 deg pitch down and 30 deg yaw."""
        frame = self.test_frame.copy()
        
        pitch = apply_angle_pitch_down_30deg(frame)
        self.assertEqual(pitch.shape, (480, 640, 3))
        self.assertEqual(pitch.dtype, np.uint8)
        # Bottom corner should have border padding
        self.assertEqual(pitch[475, 5, 0], 40)
        
        yaw = apply_angle_yaw_30deg(frame)
        self.assertEqual(yaw.shape, (480, 640, 3))
        self.assertEqual(yaw.dtype, np.uint8)
        # Top-right corner should have border padding
        self.assertEqual(yaw[5, 630, 0], 40)

    def test_05_lighting_photometric_transformations(self):
        """Test low-light underexposure and harsh backlight flare photometric transforms."""
        frame = self.test_frame.copy()
        
        # Low light
        dim = apply_lighting_low_light(frame)
        self.assertEqual(dim.shape, (480, 640, 3))
        self.assertEqual(dim.dtype, np.uint8)
        # Mean luminance must be < 40% of original
        mean_orig = float(np.mean(frame))
        mean_dim = float(np.mean(dim))
        self.assertLess(mean_dim, mean_orig * 0.40)
        
        # Harsh backlight
        backlit = apply_lighting_harsh_backlight(frame)
        self.assertEqual(backlit.shape, (480, 640, 3))
        self.assertEqual(backlit.dtype, np.uint8)
        # Top region must have high flare luminance
        top_luminance = float(np.mean(backlit[0:50, :]))
        self.assertGreater(top_luminance, mean_orig)

    def test_06_temporal_occlusion_and_blackout_ranges(self):
        """Test temporal occlusion masking (hand drop and blackout) across time ranges."""
        dummy_frames = [self.test_frame.copy() for _ in range(40)]
        
        # Hand drop: lower 65% masked
        occ_drop, win_drop = apply_temporal_occlusion(dummy_frames, "hand_drop", occ_start_frac=0.30, occ_duration_frames=10)
        s, e = win_drop
        self.assertEqual(len(occ_drop), 40)
        self.assertEqual(e - s, 10)
        # Frame before occlusion should be untouched
        self.assertGreater(float(np.mean(occ_drop[s - 1])), 100.0)
        # Frame during occlusion has lower region blacked out
        h = occ_drop[s].shape[0]
        y_split = int(h * 0.35)
        self.assertEqual(np.sum(occ_drop[s][y_split:, :]), 0)
        
        # Full blackout: entire frame zero
        occ_black, win_black = apply_temporal_occlusion(dummy_frames, "blackout", occ_start_frac=0.30, occ_duration_frames=10)
        s2, e2 = win_black
        self.assertEqual(np.sum(occ_black[s2]), 0)

    def test_07_temporal_speed_adjustments(self):
        """Test rapid speedup, prolonged slowdown, and standard speed temporal sampling."""
        dummy_frames = [self.test_frame.copy() for _ in range(50)]
        
        # Standard: unchanged
        std = apply_temporal_speed(dummy_frames, "standard")
        self.assertEqual(len(std), 50)
        
        # Rapid: subsampled (~1.75x speedup -> fewer frames)
        rapid = apply_temporal_speed(dummy_frames, "rapid")
        self.assertLess(len(rapid), len(dummy_frames))
        self.assertGreaterEqual(len(rapid), 16)
        
        # Prolonged: duplicated (~0.65x slowdown -> more frames)
        prolonged = apply_temporal_speed(dummy_frames, "prolonged")
        self.assertGreater(len(prolonged), len(dummy_frames))

    def test_08_resting_sequence_and_idle_decay(self):
        """Test appending resting frames and idle decay latency tracking."""
        dummy_frames = [self.test_frame.copy() for _ in range(30)]
        rest_seq, rest_start = create_resting_sequence(dummy_frames, rest_frames=25)
        
        self.assertEqual(len(rest_seq), 55)
        self.assertEqual(rest_start, 30)
        
        # Verify lower area is zeroed in resting frames
        h = rest_seq[35].shape[0]
        y_split = int(h * 0.30)
        self.assertEqual(np.sum(rest_seq[35][y_split:, :]), 0)

    def test_09_consecutive_signing_buffer_clearing(self):
        """Test chaining consecutive gestures with inter-sign pause."""
        frames_a = [self.test_frame.copy() for _ in range(25)]
        frames_b = [self.test_frame.copy() for _ in range(25)]
        
        chain, pause_win = create_consecutive_sequence(frames_a, frames_b, pause_frames=16)
        self.assertEqual(len(chain), 25 + 16 + 25)
        self.assertEqual(pause_win, (25, 41))
        # Pause frames must be blank/neutral
        self.assertEqual(chain[30][0, 0, 0], 40)

    def test_10_false_positive_idle_baseline(self):
        """Test that pure non-signing background produces 0.0 false positive rate and IDLE status."""
        evaluator = CameraRuntimeEvaluator(scaler_path=self.scaler_path, model_path=self.model_path)
        idle_frames = create_idle_baseline_sequence(num_frames=30)
        
        res = evaluator.evaluate_frames(idle_frames, ground_truth="IDLE", is_idle_test=True)
        evaluator.close()
        
        self.assertEqual(res["predicted_sign"], "IDLE")
        self.assertEqual(res["false_positive_rate"], 0.0)
        self.assertEqual(res["status"], "IDLE_CONFIRMED")
        self.assertEqual(res["stability_transitions"], 0)

    def test_11_reference_clip_pool_integrity(self):
        """Test reference clip pool contains valid one-handed and two-handed test clips."""
        pool = build_reference_clip_pool()
        self.assertGreaterEqual(len(pool), 8)
        
        one_handed = [k for k, v in pool.items() if v["type"] == "one_handed"]
        two_handed = [k for k, v in pool.items() if v["type"] == "two_handed"]
        
        self.assertIn("now", one_handed)
        self.assertIn("deaf", one_handed)
        self.assertIn("book", two_handed)
        self.assertIn("table", two_handed)
        
        # Verify video paths exist
        for sign, info in pool.items():
            path = str(self.repo_root / info["video_path"])
            self.assertTrue(os.path.exists(path), f"Missing clip for {sign}: {path}")

    def test_12_csv_reports_schema_integrity(self):
        """Test that generated Phase 6E CSV reports match exact required schemas and contain zero NaNs."""
        cond_csv = self.repo_root / "reports" / "phase6e_condition_results.csv"
        stab_csv = self.repo_root / "reports" / "phase6e_runtime_stability.csv"
        
        self.assertTrue(cond_csv.exists(), f"Missing condition CSV at {cond_csv}")
        self.assertTrue(stab_csv.exists(), f"Missing stability CSV at {stab_csv}")
        
        df_cond = pd.read_csv(cond_csv)
        df_stab = pd.read_csv(stab_csv)
        
        # Verify required columns for condition_results.csv
        required_cond_cols = [
            "condition", "sequence_id", "ground_truth", "predicted_sign",
            "confidence", "time_to_first_stable", "stability_transitions",
            "recovery_time_sec", "status"
        ]
        for col in required_cond_cols:
            self.assertIn(col, df_cond.columns, f"Missing required column in condition results CSV: {col}")
            
        # Verify required columns for runtime_stability.csv
        required_stab_cols = [
            "temporal_continuity", "transitions", "idle_decay_latency", "false_positive_rate"
        ]
        for col in required_stab_cols:
            self.assertIn(col, df_stab.columns, f"Missing required column in runtime stability CSV: {col}")
            
        # Verify non-empty and zero NaNs
        self.assertGreaterEqual(len(df_cond), 25, "Expected at least 25 condition rows")
        self.assertGreaterEqual(len(df_stab), 20, "Expected at least 20 stability rows")
        self.assertEqual(df_cond.isna().sum().sum(), 0, "Condition CSV must contain zero NaNs")
        self.assertEqual(df_stab.isna().sum().sum(), 0, "Stability CSV must contain zero NaNs")


if __name__ == "__main__":
    unittest.main()
