"""
Unit test suite verifying the official Track 02 prediction interface for EchoGesture.
Tests predict(video_path: str) -> dict, EchoGesturePredictor, CLI, and edge cases.
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predict import predict, EchoGesturePredictor, OFFICIAL_VOCABULARY, VALID_PREDICTIONS


class TestPredictInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = REPO_ROOT
        cls.sample_video_chair = str(cls.repo_root / "data" / "raw" / "ms_asl" / "msasl_test_1732_nHnqAu_1K-c.mp4")
        cls.sample_video_candy = str(cls.repo_root / "data" / "raw" / "ms_asl" / "msasl_pilot_test_16_candy_HuViTIjghKk.mp4")

        # Verify test videos exist
        assert os.path.exists(cls.sample_video_chair), f"Missing test video: {cls.sample_video_chair}"
        assert os.path.exists(cls.sample_video_candy), f"Missing test video: {cls.sample_video_candy}"

        # Create temporary zero-hand (black frames) mock video
        cls.zero_hand_video = os.path.join(tempfile.gettempdir(), "test_echogesture_zero_hand.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cls.zero_hand_video, fourcc, 30.0, (640, 480))
        for _ in range(15):
            writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.release()

        # Create temporary 0-byte corrupt file
        cls.empty_corrupt_video = os.path.join(tempfile.gettempdir(), "test_echogesture_empty.mp4")
        with open(cls.empty_corrupt_video, "wb") as f:
            pass

        # Create temporary non-video text file with .mp4 extension
        cls.garbage_corrupt_video = os.path.join(tempfile.gettempdir(), "test_echogesture_garbage.mp4")
        with open(cls.garbage_corrupt_video, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_123456789")

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary test files
        for p in [cls.zero_hand_video, cls.empty_corrupt_video, cls.garbage_corrupt_video]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_01_schema_validation(self):
        """Verify schema of returned dictionary: all keys present and correct types."""
        res = predict(self.sample_video_chair)

        self.assertIsInstance(res, dict)
        required_keys = {"clip_id", "prediction", "top5", "confidence", "latency_ms"}
        self.assertEqual(set(res.keys()), required_keys, f"Keys mismatch. Expected {required_keys}, got {set(res.keys())}")

        # Clip ID
        self.assertIsInstance(res["clip_id"], str)
        self.assertEqual(res["clip_id"], "msasl_test_1732_nHnqAu_1K-c")

        # Prediction
        self.assertIsInstance(res["prediction"], str)
        self.assertTrue(len(res["prediction"]) > 0)

        # Top 5
        self.assertIsInstance(res["top5"], list)
        self.assertEqual(len(res["top5"]), 5)
        for item in res["top5"]:
            self.assertIsInstance(item, str)

        # Confidence
        self.assertIsInstance(res["confidence"], float)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        # Latency
        self.assertIsInstance(res["latency_ms"], float)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_02_prediction_uppercase_and_in_vocabulary(self):
        """Verify predicted gloss is strictly uppercase and belongs to the official 40-sign vocabulary."""
        res = predict(self.sample_video_chair)
        pred = res["prediction"]

        self.assertTrue(pred.isupper(), f"Prediction '{pred}' is not uppercase")
        self.assertIn(pred, OFFICIAL_VOCABULARY, f"Prediction '{pred}' not in official 40-sign vocabulary")
        self.assertEqual(pred, "CHAIR")

        # Test on another distinct clip
        res_candy = predict(self.sample_video_candy)
        pred_candy = res_candy["prediction"]
        self.assertTrue(pred_candy.isupper(), f"Prediction '{pred_candy}' is not uppercase")
        self.assertIn(pred_candy, OFFICIAL_VOCABULARY, f"Prediction '{pred_candy}' not in vocabulary")
        self.assertEqual(pred_candy, "CANDY")

    def test_03_top5_length_ordering_and_uniqueness(self):
        """Verify top-5 candidate rankings: exactly 5 items, uppercase, in vocabulary, ordered by confidence."""
        res = predict(self.sample_video_chair)
        top5 = res["top5"]

        self.assertEqual(len(top5), 5, f"Expected exactly 5 candidates in top-5, got {len(top5)}")

        # Candidate names must be uppercase and valid
        for rank_idx, candidate in enumerate(top5):
            self.assertTrue(candidate.isupper(), f"Candidate '{candidate}' at rank {rank_idx+1} is not uppercase")
            self.assertIn(candidate, OFFICIAL_VOCABULARY, f"Candidate '{candidate}' not in official vocabulary")

        # Top-1 candidate must match the top prediction
        self.assertEqual(top5[0], res["prediction"], f"Top-1 candidate {top5[0]} does not match prediction {res['prediction']}")

        # Top 5 must be distinct unique classes
        self.assertEqual(len(set(top5)), 5, f"Top-5 list contains duplicate candidates: {top5}")

    def test_04_non_existent_path_raises_filenotfound(self):
        """Verify that passing non-existent video paths raises FileNotFoundError with a clear message."""
        non_existent_str = "data/raw/ms_asl/non_existent_video_file_99999.mp4"
        with self.assertRaises(FileNotFoundError) as ctx:
            predict(non_existent_str)
        self.assertIn("non_existent_video_file_99999.mp4", str(ctx.exception))

        # Test with pathlib.Path
        non_existent_path = Path("path/to/missing_video_123.mp4")
        with self.assertRaises(FileNotFoundError) as ctx:
            predict(non_existent_path)
        self.assertIn("missing_video_123.mp4", str(ctx.exception))

    def test_05_corrupt_or_empty_video_raises_value_error(self):
        """Verify empty (0-byte) or corrupt files raise ValueError."""
        # 1. 0-byte file
        with self.assertRaises(ValueError) as ctx:
            predict(self.empty_corrupt_video)
        self.assertIn("Empty or corrupted video file", str(ctx.exception))

        # 2. Garbage non-video file
        with self.assertRaises(ValueError) as ctx:
            predict(self.garbage_corrupt_video)
        self.assertIn("Empty or corrupted video file", str(ctx.exception))

    def test_06_zero_hand_mock_video_behavior(self):
        """Verify videos where no hands are detected return fallback IDLE prediction with 0.0 confidence."""
        res = predict(self.zero_hand_video)

        self.assertIsInstance(res, dict)
        self.assertEqual(res["clip_id"], "test_echogesture_zero_hand")
        self.assertEqual(res["prediction"], "IDLE")
        self.assertEqual(res["confidence"], 0.0)
        self.assertGreater(res["latency_ms"], 0.0)
        self.assertEqual(res["top5"], ["IDLE"] * 5)
        self.assertIn(res["prediction"], VALID_PREDICTIONS)

    def test_07_pathlib_path_support(self):
        """Verify predict() accepts pathlib.Path objects as valid input."""
        res = predict(Path(self.sample_video_chair))
        self.assertEqual(res["prediction"], "CHAIR")
        self.assertEqual(res["clip_id"], "msasl_test_1732_nHnqAu_1K-c")

    def test_08_cli_interface_success_and_error(self):
        """Verify CLI execution (python predict.py --video <path>)."""
        # 1. Success execution
        cmd_success = [sys.executable, str(self.repo_root / "predict.py"), "--video", self.sample_video_chair]
        proc = subprocess.run(cmd_success, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI failed: {proc.stderr}")

        # Parse JSON output from stdout
        # Filter any C++ warnings if present
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith(("{", "}", '"', "]", "["))]
        json_str = "\n".join(stdout_lines)
        cli_result = json.loads(json_str)

        self.assertEqual(cli_result["clip_id"], "msasl_test_1732_nHnqAu_1K-c")
        self.assertEqual(cli_result["prediction"], "CHAIR")
        self.assertEqual(len(cli_result["top5"]), 5)
        self.assertGreater(cli_result["confidence"], 0.9)

        # 2. Error on missing file
        cmd_fail = [sys.executable, str(self.repo_root / "predict.py"), "--video", "missing_clip.mp4"]
        proc_fail = subprocess.run(cmd_fail, capture_output=True, text=True)
        self.assertEqual(proc_fail.returncode, 1)
        self.assertIn("Video file not found", proc_fail.stderr)

    def test_09_context_manager_and_resource_lifecycle(self):
        """Verify EchoGesturePredictor works as context manager with clean lifecycle."""
        with EchoGesturePredictor() as predictor:
            res = predictor.predict(self.sample_video_chair)
            self.assertEqual(res["prediction"], "CHAIR")
        # Verify close has been called
        self.assertIsNone(predictor.extractor.detector.handlers if hasattr(predictor.extractor.detector, "handlers") else None)

    def test_10_latency_measurement_positive_and_realistic(self):
        """Verify latency_ms is a positive, realistic float representing processing time."""
        res = predict(self.sample_video_chair)
        self.assertIsInstance(res["latency_ms"], float)
        self.assertGreater(res["latency_ms"], 10.0, "Latency should reflect real video decoding and feature extraction")


if __name__ == "__main__":
    unittest.main()
