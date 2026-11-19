"""
EchoGesture Phase 6E: Camera Capture & Hardware Reliability Test Suite
Comprehensive unit and integration tests verifying:
- Device probing, capability query, and auto-detection
- Decoupling of canonical ML capture from display preview mirroring
- Frame delivery timing, jitter, and dropped frame estimation
- Watchdog failure detection and consecutive error handling
- Buffer management and drain operations
- Graceful lifecycle shutdown and context manager safety
- Real hardware camera acquisition (if physical camera is available)
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import cv2

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.camera_capture import (
    CameraCapture,
    CameraConfig,
    CameraTelemetry,
    CameraDeviceInfo,
    detect_available_cameras,
    benchmark_camera,
)
from scripts.live_camera import run_live_camera


class TestCameraCaptureModule(unittest.TestCase):
    """Unit tests for src/inference/camera_capture.py and scripts/live_camera.py."""

    def setUp(self) -> None:
        # Create a synthetic 480x640 BGR test frame with an asymmetric marker
        self.test_h, self.test_w = 480, 640
        self.synthetic_frame = np.zeros((self.test_h, self.test_w, 3), dtype=np.uint8)
        # Mark left side with red square to test mirroring orientation
        self.synthetic_frame[100:150, 50:100] = [0, 0, 255]

    def test_01_camera_config_defaults(self) -> None:
        """Verify default configuration parameters."""
        cfg = CameraConfig()
        self.assertEqual(cfg.camera_index, 0)
        self.assertEqual(cfg.width, 640)
        self.assertEqual(cfg.height, 480)
        self.assertEqual(cfg.target_fps, 30)
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.buffer_size, 1)
        self.assertEqual(cfg.fourcc, "MJPG")
        self.assertTrue(cfg.mirror_display)
        self.assertEqual(cfg.max_consecutive_failures, 10)
        self.assertEqual(cfg.drop_threshold_factor, 1.8)

    def test_02_camera_config_custom(self) -> None:
        """Verify custom configuration instantiation."""
        cfg = CameraConfig(
            camera_index=1,
            width=1280,
            height=720,
            target_fps=60,
            backend="dshow",
            mirror_display=False,
            drop_threshold_factor=2.0
        )
        self.assertEqual(cfg.camera_index, 1)
        self.assertEqual(cfg.width, 1280)
        self.assertEqual(cfg.height, 720)
        self.assertEqual(cfg.target_fps, 60)
        self.assertEqual(cfg.backend, "dshow")
        self.assertFalse(cfg.mirror_display)
        self.assertEqual(cfg.drop_threshold_factor, 2.0)

    def test_03_open_invalid_index(self) -> None:
        """Verify opening an invalid camera index gracefully returns False."""
        cap = CameraCapture(CameraConfig(camera_index=9999))
        opened = cap.open()
        self.assertFalse(opened)
        self.assertFalse(cap.is_opened)
        cap.release()

    def test_04_open_invalid_file_path(self) -> None:
        """Verify opening a non-existent video path gracefully returns False."""
        cap = CameraCapture(CameraConfig(camera_index="non_existent_stream_123.mp4"))
        opened = cap.open()
        self.assertFalse(opened)
        self.assertFalse(cap.is_opened)
        cap.release()

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_05_mocked_camera_initialization(self, mock_vcap_cls: MagicMock) -> None:
        """Test successful initialization and property negotiation with mocked VideoCapture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),
        }.get(prop, 0.0)
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture(CameraConfig(camera_index=0, width=640, height=480, target_fps=30))
        self.assertTrue(cap.open())
        self.assertTrue(cap.is_opened)
        self.assertEqual(cap.actual_width, 640)
        self.assertEqual(cap.actual_height, 480)
        self.assertEqual(cap.actual_fps, 30.0)
        self.assertEqual(cap.backend_name, "DSHOW")
        self.assertEqual(cap.fourcc_str, "MJPG")

        cap.release()
        mock_cap.release.assert_called_once()
        self.assertFalse(cap.is_opened)

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_06_decoupled_canonical_vs_preview_mirroring(self, mock_vcap_cls: MagicMock) -> None:
        """
        Verify that canonical capture is strictly un-flipped for model ingestion,
        and display mirroring horizontally flips only the preview frame.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.synthetic_frame.copy())
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 480.0
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture(CameraConfig(camera_index=0, mirror_display=True))
        self.assertTrue(cap.open())

        ret, canonical, preview, telemetry = cap.read_canonical_and_preview()
        self.assertTrue(ret)
        self.assertIsNotNone(canonical)
        self.assertIsNotNone(preview)

        # 1. Canonical must match original raw frame exactly
        np.testing.assert_array_equal(canonical, self.synthetic_frame)

        # Red marker was at columns 50:100 in canonical frame
        self.assertEqual(canonical[120, 75, 2], 255)  # Red channel active
        self.assertEqual(canonical[120, self.test_w - 75, 2], 0)  # Mirrored column inactive

        # 2. Preview must be horizontally flipped: red marker now at (test_w - 1 - col)
        # Column 75 in 640-width frame flips to 640 - 1 - 75 = 564
        flipped_col = self.test_w - 1 - 75
        self.assertEqual(preview[120, flipped_col, 2], 255)  # Red active in preview
        self.assertEqual(preview[120, 75, 2], 0)  # Original position inactive

        # 3. Test explicit mirror_preview=False override
        ret_no_mir, can_no_mir, prev_no_mir, _ = cap.read_canonical_and_preview(mirror_preview=False)
        self.assertTrue(ret_no_mir)
        np.testing.assert_array_equal(prev_no_mir, self.synthetic_frame)

        cap.release()

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_07_dropped_frame_and_timing_telemetry(self, mock_vcap_cls: MagicMock) -> None:
        """Test interval calculation, jitter tracking, and dropped frame detection."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.synthetic_frame.copy())
        mock_cap.get.return_value = 30.0
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture(CameraConfig(target_fps=30, drop_threshold_factor=1.8))
        self.assertTrue(cap.open())

        # Frame 1: Initial baseline
        ret1, f1, t1 = cap.read()
        self.assertTrue(ret1)
        self.assertEqual(t1.frame_index, 1)
        self.assertFalse(t1.is_dropped_frame)

        # Simulate normal delivery for frames 2..5 (dt = 33.3ms)
        fake_time = t1.timestamp
        with patch("src.inference.camera_capture.time.perf_counter") as mock_time:
            # Normal frame
            fake_time += 0.0333
            mock_time.return_value = fake_time
            ret2, f2, t2 = cap.read()
            self.assertTrue(ret2)
            self.assertFalse(t2.is_dropped_frame)
            self.assertEqual(t2.estimated_dropped_count, 0)

            # Simulated stall: frame arrived after 100ms (expected 33.3ms -> 2 dropped frames)
            fake_time += 0.1000
            mock_time.return_value = fake_time
            ret3, f3, t3 = cap.read()
            self.assertTrue(ret3)
            self.assertTrue(t3.is_dropped_frame)
            self.assertEqual(t3.estimated_dropped_count, 2)

        metrics = cap.get_metrics()
        self.assertEqual(metrics["total_frames_read"], 3)
        self.assertEqual(metrics["estimated_dropped_frames"], 2)
        self.assertGreater(metrics["drop_rate_pct"], 0.0)

        cap.release()

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_08_consecutive_failure_watchdog(self, mock_vcap_cls: MagicMock) -> None:
        """Test failure telemetry and consecutive failure counter tracking."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Simulate camera read failure
        mock_cap.read.return_value = (False, None)
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture(CameraConfig(max_consecutive_failures=5))
        self.assertTrue(cap.open())

        for expected_fail_idx in range(1, 6):
            ret, frame, telemetry = cap.read()
            self.assertFalse(ret)
            self.assertIsNone(frame)
            self.assertEqual(telemetry.consecutive_failures, expected_fail_idx)
            self.assertTrue(telemetry.is_dropped_frame)

        metrics = cap.get_metrics()
        self.assertEqual(metrics["failed_reads"], 5)
        cap.release()

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_09_buffer_drain_operation(self, mock_vcap_cls: MagicMock) -> None:
        """Test draining stale frames using cap.grab()."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.grab.side_effect = [True, True, True, False]
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture()
        self.assertTrue(cap.open())

        drained = cap.drain_buffer(max_drain=5)
        self.assertEqual(drained, 3)
        self.assertEqual(mock_cap.grab.call_count, 4)
        cap.release()

    def test_10_context_manager_lifecycle(self) -> None:
        """Test context manager __enter__ and __exit__ guarantees resource release."""
        # Using non-existent index: should raise RuntimeError on entry
        with self.assertRaises(RuntimeError):
            with CameraCapture(CameraConfig(camera_index=9999)):
                pass

    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_11_context_manager_clean_exit(self, mock_vcap_cls: MagicMock) -> None:
        """Test that context manager releases even when exception is raised inside block."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_vcap_cls.return_value = mock_cap

        try:
            with CameraCapture(CameraConfig(camera_index=0)) as cap:
                self.assertTrue(cap.is_opened)
                raise ValueError("Simulated pipeline error")
        except ValueError:
            pass

        mock_cap.release.assert_called_once()

    @patch("src.inference.camera_capture.detect_available_cameras")
    @patch("src.inference.camera_capture.cv2.VideoCapture")
    def test_12_auto_camera_detection_selection(
        self,
        mock_vcap_cls: MagicMock,
        mock_detect: MagicMock
    ) -> None:
        """Test that camera_index='auto' selects the first readable detected camera."""
        mock_detect.return_value = [
            CameraDeviceInfo(
                index=2,
                backend="DSHOW",
                name="Camera 2 (DSHOW)",
                native_width=1280,
                native_height=720,
                native_fps=30.0,
                is_available=True,
                is_readable=True
            )
        ]
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else 720.0
        mock_cap.getBackendName.return_value = "DSHOW"
        mock_vcap_cls.return_value = mock_cap

        cap = CameraCapture(CameraConfig(camera_index="auto"))
        opened = cap.open()
        self.assertTrue(opened)
        self.assertEqual(cap._resolved_index, 2)
        mock_vcap_cls.assert_called_with(2, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.release()

    def test_13_diagnostic_overlay_rendering(self) -> None:
        """Test diagnostic overlay HUD generation produces valid image."""
        cap = CameraCapture(CameraConfig(width=640, height=480, target_fps=30))
        cap._actual_width = 640
        cap._actual_height = 480
        cap._backend_name = "DSHOW"
        cap._fourcc_str = "MJPG"

        telemetry = CameraTelemetry(
            frame_index=10,
            timestamp=time.perf_counter(),
            read_time_ms=5.2,
            delivery_interval_ms=33.1,
            instantaneous_fps=30.2,
            moving_avg_fps=29.8,
            is_dropped_frame=False,
            estimated_dropped_count=0,
            consecutive_failures=0
        )

        overlay = cap.draw_diagnostic_overlay(self.synthetic_frame, telemetry)
        self.assertEqual(overlay.shape, self.synthetic_frame.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        # Ensure top banner was drawn (banner color at (10, 10))
        self.assertEqual(list(overlay[10, 10]), [20, 20, 20])

    @patch("src.inference.camera_capture.CameraCapture.open")
    @patch("src.inference.camera_capture.CameraCapture.read")
    @patch("src.inference.camera_capture.CameraCapture.get_metrics")
    def test_14_benchmark_camera_flow(
        self,
        mock_get_metrics: MagicMock,
        mock_read: MagicMock,
        mock_open: MagicMock
    ) -> None:
        """Test automated benchmark execution workflow."""
        mock_open.return_value = True
        mock_read.return_value = (
            True,
            self.synthetic_frame.copy(),
            CameraTelemetry(1, 0.0, 5.0, 33.3, 30.0, 30.0, False, 0, 0)
        )
        mock_get_metrics.return_value = {
            "total_frames_read": 10,
            "average_fps": 30.0,
            "drop_rate_pct": 0.0
        }

        res = benchmark_camera(camera_index=0, num_frames=10, warmup_frames=2)
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("num_frames_profiled"), 10)
        self.assertEqual(res.get("warmup_frames"), 2)

    def test_15_real_hardware_camera_smoke(self) -> None:
        """Smoke test on real physical hardware (if index 0 is available on system)."""
        cap = CameraCapture(CameraConfig(camera_index=0, width=640, height=480, target_fps=30))
        opened = cap.open()
        if not opened:
            self.skipTest("Physical camera 0 not available or accessible on this host.")

        try:
            # Capture 5 frames
            for _ in range(5):
                ret, canonical, preview, telemetry = cap.read_canonical_and_preview()
                if ret and canonical is not None:
                    self.assertEqual(len(canonical.shape), 3)
                    self.assertEqual(canonical.shape[2], 3)
                    self.assertEqual(canonical.shape, preview.shape)
                    self.assertGreater(telemetry.timestamp, 0.0)

            metrics = cap.get_metrics()
            self.assertGreaterEqual(metrics["total_frames_read"], 1)
        finally:
            cap.release()
            self.assertFalse(cap.is_opened)


if __name__ == "__main__":
    unittest.main()
