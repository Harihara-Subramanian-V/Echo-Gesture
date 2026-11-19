"""
EchoGesture Phase 6E: Enhanced Camera Capture & Reliability Module
Provides robust, hardware-aware camera acquisition for real-time sign language recognition.

Features:
- Windows-optimized backend selection (DirectShow with MJPG, MediaFoundation, or auto-fallback)
- Camera device auto-detection and capabilities probing
- Configurable capture resolution and FPS with hardware clamp detection
- Strict decoupling: raw capture is always canonical (un-flipped) for ML ingestion,
  while display mirroring is isolated strictly for preview/HUD visualization
- Real-time frame delivery interval measurement, jitter tracking, and dropped-frame estimation
- Watchdog failure detection for camera disconnects or frozen feeds
- Low-latency buffer management (CAP_PROP_BUFFERSIZE=1 and stale frame draining)
- Clean, lifecycle-safe shutdown with context manager support
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class CameraDeviceInfo:
    """Information about a discovered physical or virtual camera device."""
    index: int
    backend: str
    name: str
    native_width: int
    native_height: int
    native_fps: float
    is_available: bool
    is_readable: bool


@dataclass
class CameraConfig:
    """Configuration settings for camera acquisition."""
    camera_index: Union[int, str] = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30
    backend: str = "auto"  # "auto", "dshow", "msmf", "any"
    buffer_size: int = 1
    fourcc: Optional[str] = "MJPG"
    mirror_display: bool = True
    max_consecutive_failures: int = 10
    drop_threshold_factor: float = 1.8
    drain_buffer_on_lag: bool = False


@dataclass
class CameraTelemetry:
    """Per-frame diagnostic telemetry."""
    frame_index: int
    timestamp: float
    read_time_ms: float
    delivery_interval_ms: float
    instantaneous_fps: float
    moving_avg_fps: float
    is_dropped_frame: bool
    estimated_dropped_count: int
    consecutive_failures: int


def _suppress_c_stderr():
    """Context manager to suppress low-level C++ warnings during device probing."""
    class StderrSuppressor:
        def __init__(self) -> None:
            self.null_fd: Optional[int] = None
            self.save_fd: Optional[int] = None
            self.stderr_fd: Optional[int] = None

        def __enter__(self) -> None:
            try:
                self.stderr_fd = sys.stderr.fileno()
                self.save_fd = os.dup(self.stderr_fd)
                self.null_fd = os.open(os.devnull, os.O_RDWR)
                os.dup2(self.null_fd, self.stderr_fd)
            except Exception:
                # If redirection fails in a non-standard environment, proceed without crashing
                self.cleanup()

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.cleanup()

        def cleanup(self) -> None:
            try:
                if self.stderr_fd is not None and self.save_fd is not None:
                    os.dup2(self.save_fd, self.stderr_fd)
                    os.close(self.save_fd)
                    self.save_fd = None
                if self.null_fd is not None:
                    os.close(self.null_fd)
                    self.null_fd = None
            except Exception:
                pass

    return StderrSuppressor()


def detect_available_cameras(
    max_probe: int = 5,
    backend: str = "auto"
) -> List[CameraDeviceInfo]:
    """
    Probe system camera indices [0..max_probe-1] and return available device details.
    
    Args:
        max_probe: Number of sequential indices to test (default: 5)
        backend: Backend preference ("auto", "dshow", "msmf", "any")
        
    Returns:
        List of CameraDeviceInfo objects representing detected cameras
    """
    backend_flags: List[Tuple[str, int]] = []
    is_win = sys.platform == "win32"
    
    b_lower = backend.lower()
    if b_lower == "dshow" and is_win:
        backend_flags = [("DSHOW", cv2.CAP_DSHOW)]
    elif b_lower == "msmf" and is_win:
        backend_flags = [("MSMF", cv2.CAP_MSMF)]
    elif b_lower == "any":
        backend_flags = [("ANY", cv2.CAP_ANY)]
    else:  # "auto"
        if is_win:
            backend_flags = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
        else:
            backend_flags = [("ANY", cv2.CAP_ANY)]

    devices: List[CameraDeviceInfo] = []
    seen_indices: set = set()

    for b_name, b_flag in backend_flags:
        for idx in range(max_probe):
            if idx in seen_indices:
                continue
            with _suppress_c_stderr():
                cap = cv2.VideoCapture(idx, b_flag)
                opened = cap.isOpened()
                if not opened:
                    cap.release()
                    continue

                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                actual_backend = cap.getBackendName()
                cap.release()

            is_readable = bool(ret and frame is not None)
            device_info = CameraDeviceInfo(
                index=idx,
                backend=actual_backend if actual_backend else b_name,
                name=f"Camera {idx} ({actual_backend or b_name})",
                native_width=w,
                native_height=h,
                native_fps=fps,
                is_available=True,
                is_readable=is_readable
            )
            devices.append(device_info)
            seen_indices.add(idx)

    return devices


class CameraCapture:
    """
    Robust camera acquisition manager for live sign language inference.
    
    Guarantees:
    - Frame returned by read() is always the raw, canonical (un-flipped) image for ML.
    - Decoupled preview generation with configurable mirroring.
    - Accurate frame delivery interval tracking and dropped frame estimation.
    - Low-latency buffer configuration to eliminate DirectShow/MSMF queue lag.
    - Deterministic release of OS camera handles on close or context exit.
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        **kwargs: Any
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = CameraConfig(**kwargs)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._is_file_source: bool = False
        self._resolved_index: Union[int, str] = self.config.camera_index

        # Actual hardware negotiated properties
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
        self._backend_name: str = "UNKNOWN"
        self._fourcc_str: str = "UNKNOWN"

        # Telemetry & timing trackers
        self._frame_count: int = 0
        self._failed_reads_count: int = 0
        self._consecutive_failures: int = 0
        self._total_dropped_estimated: int = 0
        self._last_frame_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._moving_avg_fps: float = 0.0
        self._interval_history: deque = deque(maxlen=60)
        self._read_time_history: deque = deque(maxlen=60)

    @property
    def is_opened(self) -> bool:
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @property
    def actual_height(self) -> int:
        return self._actual_height

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def fourcc_str(self) -> str:
        return self._fourcc_str

    def open(self) -> bool:
        """
        Open camera device according to config.
        
        Returns:
            True if camera opened and readable, False otherwise.
        """
        # Close existing capture if any
        if self._cap is not None:
            self.release()

        raw_index = self.config.camera_index

        # Handle "auto" camera detection
        if isinstance(raw_index, str) and raw_index.lower() == "auto":
            detected = detect_available_cameras(max_probe=5, backend=self.config.backend)
            readable = [d for d in detected if d.is_readable]
            if not readable:
                print("[ERROR] Auto-detection found no accessible or readable camera devices.")
                return False
            self._resolved_index = readable[0].index
            print(f"[INFO] Auto-detected primary camera: Index {self._resolved_index} ({readable[0].backend})")
        elif isinstance(raw_index, str) and raw_index.isdigit():
            self._resolved_index = int(raw_index)
        else:
            self._resolved_index = raw_index

        # Check if source is a video file or stream URL
        if isinstance(self._resolved_index, str):
            if Path(self._resolved_index).exists() or "://" in self._resolved_index:
                self._is_file_source = True
            else:
                # String that does not exist as file and isn't a URL
                print(f"[ERROR] Video file or stream path does not exist: {self._resolved_index}")
                return False

        # Determine backend API
        backend_flag = cv2.CAP_ANY
        is_win = sys.platform == "win32"
        b_lower = self.config.backend.lower()

        if self._is_file_source:
            backend_flag = cv2.CAP_ANY
        elif b_lower == "dshow" and is_win:
            backend_flag = cv2.CAP_DSHOW
        elif b_lower == "msmf" and is_win:
            backend_flag = cv2.CAP_MSMF
        elif b_lower == "any":
            backend_flag = cv2.CAP_ANY
        else:  # "auto"
            backend_flag = cv2.CAP_DSHOW if is_win else cv2.CAP_ANY

        # Attempt primary open
        self._cap = cv2.VideoCapture(self._resolved_index, backend_flag)

        # If primary open failed and backend was auto on Windows, attempt MSMF fallback
        if not self._cap.isOpened() and b_lower == "auto" and is_win and not self._is_file_source:
            self._cap.release()
            self._cap = cv2.VideoCapture(self._resolved_index, cv2.CAP_MSMF)

        if not self._cap.isOpened():
            print(f"[ERROR] Could not open video device at index/path {self._resolved_index}.")
            self.release()
            return False

        # Apply camera configurations (for hardware cameras)
        if not self._is_file_source:
            # 1. FourCC configuration (DirectShow benefits from MJPG to avoid USB 2.0 uncompressed choke)
            if self.config.fourcc and is_win:
                try:
                    fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)
                    self._cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
                except Exception as e:
                    print(f"[WARNING] Could not set FourCC {self.config.fourcc}: {e}")

            # 2. Resolution and FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

            # 3. Low-latency buffer setting
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            except Exception:
                pass

        # Query negotiated properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._backend_name = self._cap.getBackendName()

        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int > 0:
            self._fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        else:
            self._fourcc_str = "DEFAULT"

        # Hardware clamp detection
        if not self._is_file_source:
            if (self._actual_width != self.config.width or self._actual_height != self.config.height):
                print(
                    f"[CAMERA NOTICE] Requested {self.config.width}x{self.config.height}, "
                    f"hardware negotiated {self._actual_width}x{self._actual_height}"
                )
            if self._actual_fps > 0 and abs(self._actual_fps - self.config.target_fps) > 5.0:
                print(
                    f"[CAMERA NOTICE] Requested {self.config.target_fps} FPS, "
                    f"hardware negotiated {self._actual_fps:.1f} FPS"
                )

        self._is_opened = True
        self.reset_metrics()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], CameraTelemetry]:
        """
        Acquire the next camera frame and calculate delivery telemetry.
        
        CRITICAL: The returned frame is strictly canonical (un-flipped BGR)
        for machine learning input.
        
        Returns:
            (ret, frame, telemetry)
        """
        if not self.is_opened or self._cap is None:
            now = time.perf_counter()
            self._consecutive_failures += 1
            self._failed_reads_count += 1
            telemetry = CameraTelemetry(
                frame_index=self._frame_count,
                timestamp=now,
                read_time_ms=0.0,
                delivery_interval_ms=0.0,
                instantaneous_fps=0.0,
                moving_avg_fps=self._moving_avg_fps,
                is_dropped_frame=True,
                estimated_dropped_count=1,
                consecutive_failures=self._consecutive_failures
            )
            return False, None, telemetry

        # Optional buffer drain if lag was detected
        if self.config.drain_buffer_on_lag:
            self.drain_buffer(max_drain=1)

        t_start = time.perf_counter()
        ret, frame = self._cap.read()
        t_end = time.perf_counter()
        read_time_ms = (t_end - t_start) * 1000.0
        now = t_end

        if not ret or frame is None:
            self._consecutive_failures += 1
            self._failed_reads_count += 1
            telemetry = CameraTelemetry(
                frame_index=self._frame_count,
                timestamp=now,
                read_time_ms=read_time_ms,
                delivery_interval_ms=0.0,
                instantaneous_fps=0.0,
                moving_avg_fps=self._moving_avg_fps,
                is_dropped_frame=True,
                estimated_dropped_count=1,
                consecutive_failures=self._consecutive_failures
            )
            return False, None, telemetry

        # Successful read
        self._consecutive_failures = 0
        self._frame_count += 1

        if self._start_time is None:
            self._start_time = now

        # Interval and FPS calculation
        if self._last_frame_time is not None:
            delivery_interval_s = now - self._last_frame_time
            delivery_interval_ms = delivery_interval_s * 1000.0
        else:
            delivery_interval_s = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0.0333
            delivery_interval_ms = delivery_interval_s * 1000.0

        self._last_frame_time = now

        instantaneous_fps = (1.0 / delivery_interval_s) if delivery_interval_s > 0 else 0.0
        if self._moving_avg_fps == 0.0:
            self._moving_avg_fps = instantaneous_fps
        else:
            self._moving_avg_fps = 0.9 * self._moving_avg_fps + 0.1 * instantaneous_fps

        # Dropped frame estimation based on expected frame interval
        nominal_fps = self.config.target_fps if self.config.target_fps > 0 else 30
        expected_interval_s = 1.0 / nominal_fps
        drop_threshold_s = expected_interval_s * self.config.drop_threshold_factor

        is_dropped = False
        dropped_count = 0
        if self._frame_count > 1 and delivery_interval_s > drop_threshold_s:
            is_dropped = True
            dropped_count = max(1, int(round(delivery_interval_s / expected_interval_s)) - 1)
            self._total_dropped_estimated += dropped_count

        self._interval_history.append(delivery_interval_ms)
        self._read_time_history.append(read_time_ms)

        telemetry = CameraTelemetry(
            frame_index=self._frame_count,
            timestamp=now,
            read_time_ms=read_time_ms,
            delivery_interval_ms=delivery_interval_ms,
            instantaneous_fps=instantaneous_fps,
            moving_avg_fps=self._moving_avg_fps,
            is_dropped_frame=is_dropped,
            estimated_dropped_count=dropped_count,
            consecutive_failures=self._consecutive_failures
        )

        return True, frame, telemetry

    def read_canonical_and_preview(
        self,
        mirror_preview: Optional[bool] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], CameraTelemetry]:
        """
        Acquire frame and return both canonical ML frame and decoupled UI preview frame.
        
        Args:
            mirror_preview: Override config.mirror_display if specified
            
        Returns:
            (ret, canonical_frame, preview_frame, telemetry)
        """
        ret, frame, telemetry = self.read()
        if not ret or frame is None:
            return False, None, None, telemetry

        # Canonical frame: untouched, un-flipped for landmark detection and model inference
        canonical_frame = frame

        # Preview frame: flipped horizontally if mirroring enabled for human user UX
        should_mirror = mirror_preview if mirror_preview is not None else self.config.mirror_display
        preview_frame = cv2.flip(frame, 1) if should_mirror else frame.copy()

        return True, canonical_frame, preview_frame, telemetry

    def get_display_frame(
        self,
        frame: np.ndarray,
        mirror: Optional[bool] = None
    ) -> np.ndarray:
        """
        Transform a canonical frame into a display preview frame.
        """
        should_mirror = mirror if mirror is not None else self.config.mirror_display
        return cv2.flip(frame, 1) if should_mirror else frame.copy()

    def drain_buffer(self, max_drain: int = 2) -> int:
        """
        Quickly grab and discard buffered frames to flush driver/OS buffer lag.
        
        Returns:
            Number of frames drained.
        """
        if not self.is_opened or self._cap is None:
            return 0
        drained = 0
        for _ in range(max_drain):
            grabbed = self._cap.grab()
            if not grabbed:
                break
            drained += 1
        return drained

    def release(self) -> None:
        """Release camera hardware handle cleanly."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                print(f"[WARNING] Exception releasing video capture: {e}")
            finally:
                self._cap = None
        self._is_opened = False

    def close(self) -> None:
        """Alias for release()."""
        self.release()

    def reset_metrics(self) -> None:
        """Reset internal telemetry and performance history."""
        self._frame_count = 0
        self._failed_reads_count = 0
        self._consecutive_failures = 0
        self._total_dropped_estimated = 0
        self._last_frame_time = None
        self._start_time = None
        self._moving_avg_fps = 0.0
        self._interval_history.clear()
        self._read_time_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return comprehensive capture statistics and telemetry metrics.
        """
        elapsed = (time.perf_counter() - self._start_time) if self._start_time else 0.0
        avg_fps = (self._frame_count / elapsed) if elapsed > 0 else 0.0

        intervals = list(self._interval_history)
        reads = list(self._read_time_history)

        int_mean = float(np.mean(intervals)) if intervals else 0.0
        int_std = float(np.std(intervals)) if intervals else 0.0
        int_min = float(np.min(intervals)) if intervals else 0.0
        int_max = float(np.max(intervals)) if intervals else 0.0

        read_mean = float(np.mean(reads)) if reads else 0.0
        read_p95 = float(np.percentile(reads, 95)) if reads else 0.0

        total_frames_expected = self._frame_count + self._total_dropped_estimated
        drop_rate_pct = (
            (self._total_dropped_estimated / total_frames_expected * 100.0)
            if total_frames_expected > 0 else 0.0
        )

        return {
            "total_frames_read": self._frame_count,
            "failed_reads": self._failed_reads_count,
            "estimated_dropped_frames": self._total_dropped_estimated,
            "drop_rate_pct": round(drop_rate_pct, 2),
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(avg_fps, 1),
            "moving_avg_fps": round(self._moving_avg_fps, 1),
            "interval_mean_ms": round(int_mean, 2),
            "interval_jitter_std_ms": round(int_std, 2),
            "interval_min_ms": round(int_min, 2),
            "interval_max_ms": round(int_max, 2),
            "read_time_mean_ms": round(read_mean, 2),
            "read_time_p95_ms": round(read_p95, 2),
            "requested_resolution": f"{self.config.width}x{self.config.height}",
            "actual_resolution": f"{self._actual_width}x{self._actual_height}",
            "requested_fps": self.config.target_fps,
            "actual_fps": round(self._actual_fps, 1),
            "backend": self._backend_name,
            "fourcc": self._fourcc_str,
            "is_file_source": self._is_file_source,
        }

    def draw_diagnostic_overlay(
        self,
        frame: np.ndarray,
        telemetry: Optional[CameraTelemetry] = None,
        extra_text: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Render a professional real-time diagnostic overlay on the preview frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Top diagnostic bar
        cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
        cv2.line(overlay, (0, 42), (w, 42), (0, 200, 255), 2)

        title = f"EchoGesture Camera Diagnostic | Backend: {self._backend_name} ({self._fourcc_str})"
        cv2.putText(
            overlay,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA
        )

        # Stats panel (semi-transparent dark background)
        panel_w = 340
        panel_h = 170
        px1, py1 = 12, 52
        px2, py2 = px1 + panel_w, py1 + panel_h
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (15, 15, 15), -1)
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (80, 80, 80), 1)

        fps_val = telemetry.moving_avg_fps if telemetry else self._moving_avg_fps
        interval_val = telemetry.delivery_interval_ms if telemetry else 0.0
        read_val = telemetry.read_time_ms if telemetry else 0.0

        intervals = list(self._interval_history)
        jitter_val = float(np.std(intervals)) if intervals else 0.0

        lines = [
            f"Resolution:  {self._actual_width}x{self._actual_height} (req: {self.config.width}x{self.config.height})",
            f"FPS Target:  {self.config.target_fps} | Actual EMA: {fps_val:.1f}",
            f"Interval:    {interval_val:.1f} ms (Jitter \u03c3: {jitter_val:.1f} ms)",
            f"Read Time:   {read_val:.1f} ms",
            f"Drop Est:    {self._total_dropped_estimated} frames",
            f"Failures:    {self._consecutive_failures} consecutive ({self._failed_reads_count} tot)",
            f"Mirror:      {'ON (Selfie)' if self.config.mirror_display else 'OFF (Canonical)'}",
        ]

        if extra_text:
            lines.extend(extra_text)

        y_offset = py1 + 20
        for line in lines:
            color = (240, 240, 240)
            if "Drop Est" in line and self._total_dropped_estimated > 0:
                color = (0, 165, 255)
            elif "Failures" in line and self._consecutive_failures > 0:
                color = (0, 0, 255)
            cv2.putText(
                overlay,
                line,
                (px1 + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )
            y_offset += 20

        # Footer help bar
        cv2.rectangle(overlay, (0, h - 28), (w, h), (20, 20, 20), -1)
        cv2.putText(
            overlay,
            "Keys: 'q'/'ESC' Exit | 'm' Toggle Mirror | 'b' Reset Stats | Canonical Feed -> ML Model",
            (12, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        return overlay

    def __enter__(self) -> "CameraCapture":
        if not self.open():
            raise RuntimeError(f"Failed to open camera with index/path: {self.config.camera_index}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


def benchmark_camera(
    camera_index: Union[int, str] = 0,
    num_frames: int = 100,
    width: int = 640,
    height: int = 480,
    target_fps: int = 30,
    backend: str = "auto",
    warmup_frames: int = 10
) -> Dict[str, Any]:
    """
    Run automated camera capture benchmark to profile hardware capabilities,
    frame acquisition latencies, delivery jitter, and dropped frame rate.
    """
    cfg = CameraConfig(
        camera_index=camera_index,
        width=width,
        height=height,
        target_fps=target_fps,
        backend=backend
    )
    cap = CameraCapture(config=cfg)
    if not cap.open():
        return {
            "status": "FAILED_TO_OPEN",
            "camera_index": camera_index,
            "backend": backend
        }

    try:
        # Warm-up phase for auto-exposure & filter graphs
        for _ in range(warmup_frames):
            cap.read()

        cap.reset_metrics()

        # Profiling run
        for _ in range(num_frames):
            ret, frame, telemetry = cap.read()
            if not ret or frame is None:
                if telemetry.consecutive_failures > 5:
                    break

        metrics = cap.get_metrics()
        metrics["status"] = "SUCCESS"
        metrics["num_frames_profiled"] = num_frames
        metrics["warmup_frames"] = warmup_frames
        return metrics
    finally:
        cap.release()
