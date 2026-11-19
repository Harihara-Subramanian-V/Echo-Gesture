"""
EchoGesture Phase 6A: Live Pipeline Unit & Integration Test Suite
Validates camera acquisition, landmark extraction, temporal buffering,
feature preprocessing, INT8 inference, prediction stabilization, and HUD rendering.
"""

import os
import sys
import unittest
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import cv2

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.live_camera import run_live_camera
from scripts.live_inference import (
    LiveLandmarkExtractor,
    LiveTemporalPreprocessor,
    PredictionStabilizer,
    draw_hud,
    run_live_inference
)
from scripts.echogesture_runtime import EchoGestureRuntime


class TestEchoGestureLivePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.scaler_path = str(cls.repo_root / "data" / "processed" / "sequences_v3" / "scaler_params.npz")
        cls.model_path = str(cls.repo_root / "models" / "quantized" / "echogesture_exp_d_int8.tflite")
        cls.sample_video = str(cls.repo_root / "data" / "raw" / "ms_asl" / "msasl_test_1732_nHnqAu_1K-c.mp4")
        cls.latency_csv = str(cls.repo_root / "reports" / "phase6a_latency_report.csv")
        
        # Verify prerequisite model and scaler files exist
        assert os.path.exists(cls.scaler_path), f"Missing scaler at {cls.scaler_path}"
        assert os.path.exists(cls.model_path), f"Missing model at {cls.model_path}"

    def test_01_camera_failure_handling(self):
        """Test graceful failure handling for invalid camera indices or file paths."""
        # 1. Invalid index in live_camera
        res_cam = run_live_camera(camera_index=9999)
        self.assertFalse(res_cam, "Expected run_live_camera to return False for invalid camera index")

        # 2. Invalid index in live_inference
        res_infer = run_live_inference(camera_index=9999, headless=True)
        self.assertEqual(res_infer.get("status"), "CAMERA_ERROR")
        self.assertEqual(res_infer.get("frames_processed"), 0)

        # 3. Non-existent video path in live_inference
        res_file = run_live_inference(camera_index="non_existent_stream.mp4", headless=True)
        self.assertEqual(res_file.get("status"), "CAMERA_ERROR")
        self.assertEqual(res_file.get("frames_processed"), 0)

    def test_02_no_hand_scenario(self):
        """Test landmark extraction and preprocessing when no hands are in the camera frame."""
        extractor = LiveLandmarkExtractor()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        flat_134, telemetry, results = extractor.process_frame(blank_frame)
        extractor.close()
        
        self.assertEqual(flat_134.shape, (134,))
        self.assertEqual(flat_134.dtype, np.float32)
        
        # Verify coordinates and presence flags are zero
        self.assertTrue(np.all(flat_134 == 0.0), "All 134 features must be 0.0 when no hands detected")
        self.assertFalse(telemetry["left_detected"])
        self.assertFalse(telemetry["right_detected"])
        self.assertFalse(telemetry["both_detected"])
        self.assertEqual(telemetry["left_score"], 0.0)
        self.assertEqual(telemetry["right_score"], 0.0)

    def test_03_temporal_preprocessor_inactive_clamping(self):
        """Test active-frame standardization strictly preserves zero-clamping for inactive frames."""
        preprocessor = LiveTemporalPreprocessor(self.scaler_path)
        
        # Empty 64-frame window
        empty_window = np.zeros((64, 134), dtype=np.float32)
        feat_263 = preprocessor.process_window(empty_window)
        
        self.assertEqual(feat_263.shape, (64, 263))
        self.assertEqual(feat_263.dtype, np.float32)
        self.assertTrue(np.all(feat_263 == 0.0), "Preprocessed tensor for empty window must remain all zeros")
        self.assertFalse(np.isnan(feat_263).any(), "Preprocessed tensor must not contain NaNs")
        self.assertFalse(np.isinf(feat_263).any(), "Preprocessed tensor must not contain Infs")

    def test_04_inter_hand_geometry_math(self):
        """Test aspect-ratio corrected diagonal-normalized inter-hand Euclidean distance and offset."""
        w_px, h_px = 640, 480
        diag = float(np.sqrt(w_px**2 + h_px**2))
        
        # Simulate Left wrist at (0.3, 0.4) and Right wrist at (0.7, 0.6)
        x_L, y_L = 0.3, 0.4
        x_R, y_R = 0.7, 0.6
        
        dx_px = (x_R - x_L) * w_px
        dy_px = (y_R - y_L) * h_px
        dist_px = float(np.sqrt(dx_px**2 + dy_px**2))
        
        expected_dist = dist_px / diag
        expected_offset_x = dx_px / diag
        expected_offset_y = dy_px / diag
        
        self.assertGreater(expected_dist, 0.0)
        self.assertAlmostEqual(expected_dist, np.sqrt(expected_offset_x**2 + expected_offset_y**2), places=5)

    def test_05_tracking_loss_and_recovery_velocity_spikes(self):
        """Test that hand appearance, disappearance, and reappearance produce zero velocity spikes."""
        preprocessor = LiveTemporalPreprocessor(self.scaler_path)
        
        window = np.zeros((64, 134), dtype=np.float32)
        
        # Frames 10..20: Left hand active at constant coordinate 0.5
        for t in range(10, 21):
            window[t, 0:63] = 0.5
            window[t, 126] = 1.0  # Left presence flag
            
        # Frames 21..29: Hand disappears (all 0.0)
        # Frames 30..40: Hand reappears at coordinate 0.8
        for t in range(30, 41):
            window[t, 0:63] = 0.8
            window[t, 126] = 1.0  # Left presence flag
            
        feat_263 = preprocessor.process_window(window)
        
        # Left velocity features are at indices 134:197
        v_L = feat_263[:, 134:197]
        
        # Frame 10 (appearance): must be 0.0 (no previous active frame)
        self.assertTrue(np.all(v_L[10] == 0.0), "Frame 10 appearance velocity must be exactly 0.0")
        
        # Frames 11..20 (continuous): constant coordinates -> velocity must be 0.0
        self.assertTrue(np.all(v_L[11:21] == 0.0), "Constant active coordinate velocity must be 0.0")
        
        # Frame 21 (disappearance transition): must be 0.0
        self.assertTrue(np.all(v_L[21] == 0.0), "Frame 21 disappearance velocity must be exactly 0.0")
        
        # Frame 30 (reappearance transition): must be 0.0
        self.assertTrue(np.all(v_L[30] == 0.0), "Frame 30 reappearance velocity must be exactly 0.0")
        
        # Frames 31..40 (continuous): constant coordinates -> velocity must be 0.0
        self.assertTrue(np.all(v_L[31:41] == 0.0), "Continuous active velocity must be 0.0")

    def test_06_rolling_buffer_and_active_standardization(self):
        """Test rolling buffer FIFO behavior and active Z-score standardization."""
        buffer = deque(maxlen=64)
        
        # Push 80 frames
        for i in range(80):
            frame_vec = np.full(134, fill_value=float(i), dtype=np.float32)
            buffer.append(frame_vec)
            
        self.assertEqual(len(buffer), 64, "Rolling buffer must hold exactly 64 items")
        self.assertEqual(buffer[0][0], 16.0, "Oldest preserved frame index should be 16")
        self.assertEqual(buffer[-1][0], 79.0, "Newest frame index should be 79")
        
        # Test active standardization
        preprocessor = LiveTemporalPreprocessor(self.scaler_path)
        window = np.zeros((64, 134), dtype=np.float32)
        # Activate Right hand in frames 30..63
        window[30:, 63:126] = 0.5
        window[30:, 127] = 1.0  # Right presence flag
        
        feat_263 = preprocessor.process_window(window)
        
        # Inactive frames (0..29) for Right hand must remain exactly 0.0
        self.assertTrue(np.all(feat_263[0:30, 63:126] == 0.0), "Inactive frames must not be shifted by mean/std")
        
        # Active frames (30..63) for Right hand must be standardized
        expected_active = (0.5 - preprocessor.mean[63:126]) / (preprocessor.std[63:126] + 1e-6)
        expected_tiled = np.tile(expected_active, (34, 1))
        np.testing.assert_allclose(feat_263[30:, 63:126], expected_tiled, rtol=1e-4)

    def test_07_prediction_stabilization(self):
        """Test temporal hysteresis, confidence thresholding, noise rejection, and idle dropout."""
        stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35, idle_dropout_frames=12)
        
        # 1. Feed low confidence (<0.35) -> should remain UNCERTAIN / IDLE
        low_conf_pred = {"predicted_class": "book", "predicted_class_id": 5, "confidence": 0.20}
        c_name, c_id, conf = stabilizer.update(low_conf_pred, has_hands=True)
        self.assertEqual(c_name, "IDLE")
        
        # 2. Feed high confidence "book" (0.80) -> stabilizes to "book"
        high_conf_pred = {"predicted_class": "book", "predicted_class_id": 5, "confidence": 0.80}
        for _ in range(3):
            c_name, c_id, conf = stabilizer.update(high_conf_pred, has_hands=True)
        self.assertEqual(c_name, "book")
        self.assertEqual(c_id, 5)
        self.assertAlmostEqual(conf, 0.80, places=2)
        
        # 3. Single-frame noise spike: "candy" (0.40)
        noise_pred = {"predicted_class": "candy", "predicted_class_id": 12, "confidence": 0.40}
        c_name, c_id, conf = stabilizer.update(noise_pred, has_hands=True)
        self.assertEqual(c_name, "book", "Single-frame noise spike must be rejected by stabilizer")
        
        # 4. Sustained transition: feed "candy" with high confidence (0.90) for 4 frames
        candy_pred = {"predicted_class": "candy", "predicted_class_id": 12, "confidence": 0.90}
        for _ in range(4):
            c_name, c_id, conf = stabilizer.update(candy_pred, has_hands=True)
        self.assertEqual(c_name, "candy", "Stabilizer should transition to candy after sustained evidence")
        self.assertEqual(c_id, 12)
        
        # 5. Idle dropout: 12 consecutive frames with has_hands=False
        for _ in range(12):
            c_name, c_id, conf = stabilizer.update(raw_pred=None, has_hands=False)
        self.assertEqual(c_name, "IDLE", "Stabilizer must reset to IDLE after 12 idle frames")
        self.assertEqual(c_id, -1)
        self.assertEqual(conf, 0.0)

    def test_08_runtime_inference_integration(self):
        """Test inference with EchoGestureRuntime using quantized INT8 TFLite model."""
        runtime = EchoGestureRuntime(model_path=self.model_path)
        
        # Create valid (64, 263) input tensor
        synthetic_input = np.random.randn(64, 263).astype(np.float32) * 0.1
        
        pred = runtime.predict(synthetic_input)
        
        self.assertIn("predicted_class", pred)
        self.assertIn("predicted_class_id", pred)
        self.assertIn("confidence", pred)
        self.assertIn("top_5", pred)
        self.assertIn("latency_ms", pred)
        
        self.assertIn(pred["predicted_class"], list(runtime.idx2label.values()))
        self.assertEqual(len(pred["top_5"]), 5)
        # Check top-5 ordering
        top5_confs = [item["confidence"] for item in pred["top_5"]]
        self.assertEqual(top5_confs, sorted(top5_confs, reverse=True))
        self.assertAlmostEqual(pred["confidence"], top5_confs[0], places=4)
        self.assertGreater(pred["latency_ms"], 0.0)
        self.assertLess(pred["latency_ms"], 50.0, "INT8 inference latency must be well under 50ms")

    def test_09_hud_rendering(self):
        """Test live telemetry HUD overlay rendering."""
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        latencies = {
            "capture_ms": 1.2,
            "mediapipe_ms": 22.5,
            "preproc_ms": 0.15,
            "inference_ms": 0.10,
            "total_ms": 24.0
        }
        raw_pred = {
            "predicted_class": "chair",
            "predicted_class_id": 14,
            "confidence": 0.95,
            "top_5": [
                {"class_name": "chair", "confidence": 0.95},
                {"class_name": "bed", "confidence": 0.03},
                {"class_name": "table", "confidence": 0.01}
            ]
        }
        
        hud = draw_hud(
            frame=canvas,
            tracking_status="TRACKING (2 HANDS)",
            raw_pred=raw_pred,
            stabilized_class="chair",
            stabilized_conf=0.95,
            buffer_fill=64,
            fps=32.5,
            latencies=latencies,
            show_landmarks=False
        )
        
        self.assertEqual(hud.shape, (480, 640, 3))
        self.assertEqual(hud.dtype, np.uint8)
        self.assertGreater(np.sum(hud), 0, "Rendered HUD frame must not be completely black")

    def test_10_latency_benchmark_csv_integrity(self):
        """Verify that phase6a_latency_report.csv contains at least 100 valid consecutive frames."""
        self.assertTrue(os.path.exists(self.latency_csv), f"Missing latency CSV report at {self.latency_csv}")
        df = pd.read_csv(self.latency_csv)
        
        # Check rows >= 100
        self.assertGreaterEqual(len(df), 100, f"Expected >= 100 frames, found {len(df)}")
        
        required_cols = [
            "frame", "fps", "capture_ms", "mediapipe_ms",
            "preproc_ms", "inference_ms", "total_ms",
            "status", "stabilized_sign", "confidence"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column {col} in latency report")
            
        # Check for zero NaNs in numerical columns
        num_cols = ["fps", "capture_ms", "mediapipe_ms", "preproc_ms", "inference_ms", "total_ms", "confidence"]
        self.assertFalse(df[num_cols].isna().any().any(), "Latency CSV must contain no NaNs")
        
        # Verify edge latency budget: mean total latency under 40ms (> 25 FPS)
        mean_total = df["total_ms"].mean()
        mean_infer = df["inference_ms"].mean()
        self.assertLess(mean_total, 40.0, f"Mean total latency {mean_total:.2f}ms exceeds 40ms threshold")
        self.assertLess(mean_infer, 1.0, f"Mean INT8 inference latency {mean_infer:.3f}ms exceeds 1.0ms threshold")

    def test_11_incomplete_window_resampling(self):
        """Test that incomplete windows (e.g. 30 frames) are resampled to 64 frames cleanly."""
        from scripts.prepare_sequences import resample_sequence
        preprocessor = LiveTemporalPreprocessor(self.scaler_path)
        runtime = EchoGestureRuntime(model_path=self.model_path)
        
        # Simulate a 30-frame window where 20 frames have active Left hand
        window_30 = np.zeros((30, 134), dtype=np.float32)
        window_30[5:25, 0:63] = 0.5
        window_30[5:25, 126] = 1.0  # Left presence
        
        resampled_64 = resample_sequence(window_30, target_len=64)
        self.assertEqual(resampled_64.shape, (64, 134))
        
        feat_263 = preprocessor.process_window(resampled_64)
        self.assertEqual(feat_263.shape, (64, 263))
        self.assertFalse(np.isnan(feat_263).any())
        self.assertFalse(np.isinf(feat_263).any())
        
        pred = runtime.predict(feat_263)
        self.assertIn("predicted_class", pred)
        self.assertGreater(pred["confidence"], 0.0)

    def test_12_stale_prediction_uncertain_decay(self):
        """Test that consecutive uncertain predictions cause stabilized class to decay to IDLE."""
        stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35, uncertain_dropout_frames=5)
        
        # 1. Establish stabilized class "chair"
        confident_pred = {"predicted_class": "chair", "predicted_class_id": 14, "confidence": 0.85}
        for _ in range(3):
            c_name, c_id, conf = stabilizer.update(confident_pred, has_hands=True)
        self.assertEqual(c_name, "chair")
        
        # 2. Feed 5 consecutive low-confidence / uncertain predictions while hands remain visible
        uncertain_pred = {"predicted_class": "chair", "predicted_class_id": 14, "confidence": 0.20}
        for i in range(4):
            c_name, c_id, conf = stabilizer.update(uncertain_pred, has_hands=True)
            self.assertEqual(c_name, "chair", f"Expected to hold during uncertain buffer filling at step {i}")
            
        # On 5th uncertain frame, uncertain_dropout triggers
        c_name, c_id, conf = stabilizer.update(uncertain_pred, has_hands=True)
        self.assertEqual(c_name, "IDLE", "Stabilizer must decay to IDLE after 5 consecutive uncertain frames")
        self.assertEqual(c_id, -1)
        self.assertEqual(conf, 0.0)

    def test_13_two_frame_consensus_hysteresis(self):
        """Test that single-frame moderate spikes cannot flip an established class without consensus."""
        stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35)
        
        # 1. Establish "book" with 2 confident frames (total mass 1.60)
        for _ in range(2):
            stabilizer.update({"predicted_class": "book", "predicted_class_id": 5, "confidence": 0.80}, has_hands=True)
        self.assertEqual(stabilizer.current_stabilized_class, "book")
        
        # 2. Feed a single-frame moderate spike of "candy" (0.55 confidence)
        # Even if candy were to spike, a single frame cannot flip an established class
        c_name, c_id, conf = stabilizer.update({"predicted_class": "candy", "predicted_class_id": 12, "confidence": 0.55}, has_hands=True)
        self.assertEqual(c_name, "book", "Single-frame spike must not flip established class 'book'")
        
        # 3. Feed 2 more confirming frames for "candy" (0.85 confidence each) -> candy takes leading score and has >= 2 votes
        stabilizer.update({"predicted_class": "candy", "predicted_class_id": 12, "confidence": 0.85}, has_hands=True)
        c_name, c_id, conf = stabilizer.update({"predicted_class": "candy", "predicted_class_id": 12, "confidence": 0.85}, has_hands=True)
        self.assertEqual(c_name, "candy", "Confirming consensus votes must trigger transition to 'candy'")

    def test_14_unflipped_feature_extraction_parity(self):
        """Test that LiveLandmarkExtractor on canonical raw frames matches offline feature tensors."""
        offline_npz = Path(self.repo_root) / "data" / "processed" / "landmarks_v3" / "msasl_test_1780_otTiJBUrJmI.npz"
        if not offline_npz.exists():
            self.skipTest("Offline Phase 3 NPZ not found")
            
        offline_134 = np.load(offline_npz)["flat_features"]
        video_path = str(Path(self.repo_root) / "data" / "raw" / "ms_asl" / "msasl_test_1780_otTiJBUrJmI.mp4")
        
        cap = cv2.VideoCapture(video_path)
        extractor = LiveLandmarkExtractor()
        live_frames = []
        for _ in range(20):  # Check first 20 frames
            ret, frame = cap.read()
            if not ret:
                break
            f134, _, _ = extractor.process_frame(frame)
            live_frames.append(f134)
        cap.release()
        extractor.close()
        
        live_134 = np.array(live_frames, dtype=np.float32)
        diff = np.abs(offline_134[:len(live_134)] - live_134)
        
        # Verify mean absolute difference is sub-millimeter (< 0.005)
        mean_diff = float(np.mean(diff))
        self.assertLess(mean_diff, 0.005, f"Expected mean diff < 0.005, got {mean_diff:.5f}")
        
        # Coordinate match rate within 1% of screen dimension
        match_rate = float(np.mean(diff < 1e-2) * 100.0)
        self.assertGreater(match_rate, 95.0, f"Expected >95% coordinate agreement within 0.01, got {match_rate:.2f}%")


if __name__ == "__main__":
    unittest.main()
