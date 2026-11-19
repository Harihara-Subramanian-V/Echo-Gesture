"""
EchoGesture Phase 6E: Live Camera Acquisition & Hardware Diagnostic Utility
Robust camera verification, device auto-detection, and capture reliability tool.

Features:
- Windows-optimized capture using DirectShow (MJPG) and MediaFoundation
- Automatic camera enumeration and capability probing (--probe)
- Hardware capture benchmarking (--benchmark)
- Strict decoupling: raw frame is always canonical un-flipped for ML ingestion;
  display mirroring is isolated strictly for preview/UX
- Real-time diagnostics: frame delivery interval, jitter (std dev), dropped frame tracking
- Clean shutdown on 'q', ESC, or interrupt signal
- Configurable headless and scripted execution (--headless, --max-frames)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

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


def run_live_camera(
    camera_index: Union[int, str] = 0,
    width: int = 640,
    height: int = 480,
    fps_target: int = 30,
    backend: str = "auto",
    mirror_display: bool = True,
    diagnostics: bool = True,
    max_frames: Optional[int] = None,
    headless: bool = False
) -> bool:
    """
    Open camera feed and display live video with comprehensive diagnostic telemetry.
    
    Args:
        camera_index: Device index (int or str), video file path, or "auto"
        width: Requested frame width
        height: Requested frame height
        fps_target: Target FPS
        backend: "auto", "dshow", "msmf", or "any"
        mirror_display: Enable mirror preview horizontally for selfie view
        diagnostics: Display rich real-time diagnostic overlay
        max_frames: Optional limit of frames before automatic clean exit
        headless: Run without GUI display window
        
    Returns:
        True if camera opened and stream ran cleanly, False if camera failed to open.
    """
    print("=== EchoGesture Live Camera & Hardware Diagnostics ===")
    print(f"Device Target:     {camera_index} ({width}x{height} @ {fps_target} FPS)")
    print(f"Backend API:       {backend.upper()}")
    print(f"Mirror Display:    {mirror_display} (Model ingestion is always canonical un-flipped)")
    print(f"Diagnostic Mode:   {diagnostics}")
    print(f"Headless Mode:     {headless}")
    print("------------------------------------------------------")

    config = CameraConfig(
        camera_index=camera_index,
        width=width,
        height=height,
        target_fps=fps_target,
        backend=backend,
        mirror_display=mirror_display,
        buffer_size=1
    )

    cap = CameraCapture(config=config)
    if not cap.open():
        print(f"\n[ERROR] Could not open video device at camera index/path: {camera_index}")
        print("Possible causes:")
        print("  1. No webcam connected or device disabled in Windows Device Manager.")
        print("  2. Camera is in use by another application (e.g. Teams, Zoom, Browser).")
        print("  3. Camera index is different (run with --probe to detect valid indices).")
        print("\nExiting gracefully.")
        return False

    print(
        f"[SUCCESS] Camera opened: {cap.actual_width}x{cap.actual_height} "
        f"@ {cap.actual_fps:.1f} FPS via {cap.backend_name} ({cap.fourcc_str})"
    )
    if not headless:
        print("Controls: 'q'/ESC to exit | 'm' toggle mirror | 'd' toggle diagnostic HUD | 'b' reset stats\n")

    window_name = "EchoGesture - Camera Feed (Press 'q' or ESC to exit)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    active_mirror = mirror_display
    show_diagnostics = diagnostics
    frame_counter = 0

    try:
        while True:
            # Acquire frame: canonical is strictly untouched, preview is for display
            ret, canonical_frame, preview_frame, telemetry = cap.read_canonical_and_preview(
                mirror_preview=active_mirror
            )

            if not ret or preview_frame is None:
                if telemetry.consecutive_failures >= cap.config.max_consecutive_failures:
                    print(f"\n[ERROR] Camera disconnected or failed after {telemetry.consecutive_failures} consecutive errors.")
                    break
                continue

            frame_counter += 1

            if not headless:
                if show_diagnostics:
                    display_frame = cap.draw_diagnostic_overlay(
                        frame=preview_frame,
                        telemetry=telemetry
                    )
                else:
                    display_frame = preview_frame.copy()
                    # Minimal overlay
                    cv2.putText(
                        display_frame,
                        f"Live Camera | {cap.actual_width}x{cap.actual_height} | FPS: {telemetry.moving_avg_fps:.1f}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 is ESC
                    print("\nExit signal received from user.")
                    break
                elif key == ord('m'):
                    active_mirror = not active_mirror
                    print(f"Display mirror preview: {active_mirror}")
                elif key == ord('d'):
                    show_diagnostics = not show_diagnostics
                    print(f"Diagnostic overlay: {show_diagnostics}")
                elif key == ord('b'):
                    cap.reset_metrics()
                    print("Performance metrics reset.")

            if max_frames and frame_counter >= max_frames:
                print(f"Reached max_frames limit ({max_frames}). Exiting loop.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
    finally:
        metrics = cap.get_metrics()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

        print("\n=== Camera Capture Session Summary ===")
        print(f"Total Frames Captured:     {metrics['total_frames_read']}")
        print(f"Failed Grab Attempts:      {metrics['failed_reads']}")
        print(f"Estimated Dropped Frames:  {metrics['estimated_dropped_frames']} ({metrics['drop_rate_pct']}%)")
        print(f"Elapsed Time:              {metrics['elapsed_seconds']} s")
        print(f"Average Measured FPS:      {metrics['average_fps']}")
        print(f"Frame Interval (mean):     {metrics['interval_mean_ms']} ms (jitter \u03c3: {metrics['interval_jitter_std_ms']} ms)")
        print(f"Frame Read Time (mean):    {metrics['read_time_mean_ms']} ms (p95: {metrics['read_time_p95_ms']} ms)")
        print(f"Device Backend & FourCC:   {metrics['backend']} / {metrics['fourcc']}")
        print("Camera released and resources closed cleanly.")

    return True


def probe_and_print_cameras(max_probe: int = 5, backend: str = "auto") -> None:
    """Scan and print connected camera devices."""
    print("=== Scanning Available Camera Devices ===")
    devices = detect_available_cameras(max_probe=max_probe, backend=backend)
    if not devices:
        print("No camera devices detected.")
        return

    print(f"Found {len(devices)} device(s):")
    for dev in devices:
        status = "READABLE" if dev.is_readable else "UNREADABLE / IN-USE"
        print(f"  [{dev.index}] {dev.name}: {dev.native_width}x{dev.native_height} @ {dev.native_fps} FPS - {status}")
    print()


def run_benchmark(
    camera_index: Union[int, str] = 0,
    frames: int = 100,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    backend: str = "auto"
) -> None:
    """Run camera acquisition benchmark and print detailed analysis."""
    print(f"=== Running Camera Hardware Benchmark ({frames} frames) ===")
    results = benchmark_camera(
        camera_index=camera_index,
        num_frames=frames,
        width=width,
        height=height,
        target_fps=fps,
        backend=backend
    )

    if results.get("status") != "SUCCESS":
        print(f"[ERROR] Benchmark failed: {results}")
        return

    print("\nBenchmark Results:")
    print(f"  Camera Index:             {camera_index}")
    print(f"  Backend:                  {results['backend']} ({results['fourcc']})")
    print(f"  Requested Resolution:     {results['requested_resolution']}")
    print(f"  Negotiated Resolution:    {results['actual_resolution']}")
    print(f"  Requested FPS:            {results['requested_fps']}")
    print(f"  Achieved Average FPS:     {results['average_fps']}")
    print(f"  Inter-frame Interval:     {results['interval_mean_ms']} ms (min: {results['interval_min_ms']} ms, max: {results['interval_max_ms']} ms)")
    print(f"  Interval Jitter (\u03c3):        {results['interval_jitter_std_ms']} ms")
    print(f"  Frame Read Latency (mean):{results['read_time_mean_ms']} ms (p95: {results['read_time_p95_ms']} ms)")
    print(f"  Estimated Dropped Frames: {results['estimated_dropped_frames']} ({results['drop_rate_pct']}%)")
    print(f"  Total Frames Profiled:    {results['num_frames_profiled']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Live Camera & Hardware Diagnostic Utility")
    parser.add_argument("--camera", type=str, default="0", help="Camera index, video path, or 'auto' (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "dshow", "msmf", "any"],
                        help="Camera backend API (default: auto)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable display preview mirroring")
    parser.add_argument("--minimal", action="store_true", help="Use minimal HUD instead of diagnostic panel")
    parser.add_argument("--probe", action="store_true", help="Scan and list all detected camera devices and exit")
    parser.add_argument("--benchmark", action="store_true", help="Run acquisition benchmark on camera and exit")
    parser.add_argument("--bench-frames", type=int, default=100, help="Number of benchmark frames (default: 100)")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after processing N frames")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    # Parse camera argument: integer if digit
    cam_arg: Union[int, str] = int(args.camera) if args.camera.isdigit() else args.camera

    if args.probe:
        probe_and_print_cameras(backend=args.backend)
    elif args.benchmark:
        run_benchmark(
            camera_index=cam_arg,
            frames=args.bench_frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=args.backend
        )
    else:
        run_live_camera(
            camera_index=cam_arg,
            width=args.width,
            height=args.height,
            fps_target=args.fps,
            backend=args.backend,
            mirror_display=not args.no_mirror,
            diagnostics=not args.minimal,
            max_frames=args.max_frames,
            headless=args.headless
        )
