"""
EchoGesture Phase 6E: High-Precision Runtime Latency Profiler
Profiles stage-by-stage computational latency, capture frame rate, effective pipeline throughput,
and latency distributions on physical camera streams or video fallbacks.

Pipeline Stages Profiled:
1. Camera capture latency (ms) - Physical sensor read & driver synchronization
2. MediaPipe landmark extraction latency (ms) - Detection, tracking & 134-D assembly
3. Temporal preprocessing & velocity latency (ms) - Active Z-score & 263-D velocity
4. INT8 TFLite model-only inference latency (ms) - Pure interpreter.invoke()
5. Runtime wrapper latency (ms) - Quantization + invoke + dequantization + top-5
6. Temporal stabilization & hysteresis latency (ms) - Sliding window voting
7. Total end-to-end latency (ms) - Full frame loop time
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd

# Suppress verbose TensorFlow / absl logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.live_inference import LiveLandmarkExtractor, LiveTemporalPreprocessor, PredictionStabilizer
from scripts.prepare_sequences import resample_sequence


def compute_distribution_metrics(values: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Compute standard statistical distribution metrics for an array of latency values."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def profile_pipeline(
    source: Union[int, str] = 0,
    target_frames: int = 350,
    warmup_frames: int = 20,
    width: int = 640,
    height: int = 480,
    target_fps: int = 30,
    active_inference: bool = True,
    num_threads: int = 1,
    output_csv: Optional[str] = None,
    output_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Profile high-precision stage-by-stage latency of the EchoGesture live pipeline.

    Args:
        source: Camera index (int) or path to video file (str)
        target_frames: Total number of frames to profile (after warmup)
        warmup_frames: Number of initial frames to discard for CPU/cache warmup
        width: Requested camera frame width
        height: Requested camera frame height
        target_fps: Target camera frame rate
        active_inference: If True, execute temporal preprocessor and INT8 model on every frame
                          to profile peak active recognition load
        num_threads: Number of CPU threads for TFLite interpreter
        output_csv: Path to save per-frame latency CSV report
        output_summary: Path to save JSON summary metrics

    Returns:
        Structured dictionary containing benchmark results and distributions
    """
    scaler_path = str(REPO_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz")
    model_path = str(REPO_ROOT / "models" / "quantized" / "echogesture_exp_d_int8.tflite")

    print("=" * 72)
    print(" EchoGesture Phase 6E: High-Precision Runtime Latency Profiler")
    print("=" * 72)
    print(f"Source:                {source} (Type: {'Webcam / Live Stream' if isinstance(source, int) else 'Video File'})")
    print(f"Target Resolution:     {width}x{height} @ {target_fps} FPS (Budget: {1000.0/target_fps:.2f} ms/frame)")
    print(f"Target Frames:         {target_frames} (+ {warmup_frames} warmup frames)")
    print(f"Peak Active Mode:      {active_inference} (Executes full preproc + INT8 model every frame)")
    print(f"TFLite Threads:        {num_threads}")
    print(f"Model Path:            {model_path}")
    print(f"Scaler Path:           {scaler_path}")
    print("=" * 72)

    # 1. Initialize Pipeline Components
    print("\n[1/4] Initializing pipeline components...")
    extractor = LiveLandmarkExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path, num_threads=num_threads)
    stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35)

    buffer_134 = deque(maxlen=64)

    # 2. Open Video Source
    print(f"[2/4] Opening video source: {source}...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        error_msg = f"Failed to open video source '{source}'."
        print(f"[ERROR] {error_msg}")
        extractor.close()
        raise RuntimeError(error_msg)

    # Configure hardware capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)
    backend_name = cap.getBackendName()
    print(f"[INFO] Video backend: {backend_name}, Actual resolution: {actual_w}x{actual_h}, Prop FPS: {actual_fps_prop}")

    # 3. Warm-up Phase
    print(f"[3/4] Executing {warmup_frames} warm-up frames to prime CPU instruction cache...")
    for w_i in range(warmup_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            # If video ends during warmup, loop back
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frames during warmup phase.")
        flat_134, telemetry, _ = extractor.process_frame(frame)
        buffer_134.append(flat_134)
        if len(buffer_134) >= 16:
            window_arr = np.array(buffer_134, dtype=np.float32)
            resampled = resample_sequence(window_arr, 64) if len(buffer_134) < 64 else window_arr
            feat = preprocessor.process_window(resampled)
            pred = runtime.predict(feat)
            stabilizer.update(pred, True)

    print("[SUCCESS] Warm-up complete. Starting high-precision profiling loop...\n")

    # 4. Profiling Execution Loop
    records: List[Dict[str, Any]] = []
    t_bench_start = time.perf_counter()
    prev_cap_timestamp = t_bench_start

    frame_budget_ms = 1000.0 / target_fps
    over_budget_count = 0
    consecutive_over_budget = 0
    max_consecutive_over_budget = 0

    try:
        for frame_idx in range(1, target_frames + 1):
            t_loop_start = time.perf_counter()

            # --- Stage 1: Camera Capture ---
            t_cap_start = time.perf_counter()
            ret, frame = cap.read()
            t_cap_end = time.perf_counter()
            capture_ms = (t_cap_end - t_cap_start) * 1000.0

            if not ret or frame is None:
                # Video file reached end -> rewind if video file to finish target_frames
                if not isinstance(source, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t_cap_start = time.perf_counter()
                    ret, frame = cap.read()
                    t_cap_end = time.perf_counter()
                    capture_ms = (t_cap_end - t_cap_start) * 1000.0
                    if not ret or frame is None:
                        print(f"[WARNING] End of video reached at frame {frame_idx}.")
                        break
                else:
                    print(f"[WARNING] Live camera frame grab failed at frame {frame_idx}.")
                    break

            capture_interval_ms = (t_cap_end - prev_cap_timestamp) * 1000.0
            prev_cap_timestamp = t_cap_end

            # --- Stage 2: MediaPipe Landmark Extraction ---
            t_mp_start = time.perf_counter()
            flat_134, telemetry, mp_res = extractor.process_frame(frame)
            t_mp_end = time.perf_counter()
            mediapipe_ms = (t_mp_end - t_mp_start) * 1000.0

            buffer_134.append(flat_134)
            buf_len = len(buffer_134)
            has_hands = bool(telemetry["left_detected"] or telemetry["right_detected"])

            # Check active frames in current temporal window
            window_arr = np.array(buffer_134, dtype=np.float32)
            active_frames_count = int(np.sum((window_arr[:, 126] > 0.5) | (window_arr[:, 127] > 0.5)))

            # Determine whether to execute inference
            should_infer = active_inference or (buf_len >= 16 and active_frames_count >= 12)

            preproc_ms = 0.0
            model_invoke_ms = 0.0
            runtime_wrapper_ms = 0.0
            pred_class = "IDLE"
            pred_conf = 0.0
            raw_pred: Optional[Dict[str, Any]] = None

            if should_infer:
                # --- Stage 3: Temporal Preprocessing & 263-D Velocity ---
                t_prep_start = time.perf_counter()
                if buf_len < 64:
                    resampled_134 = resample_sequence(window_arr, target_len=64)
                else:
                    resampled_134 = window_arr

                feat_263 = preprocessor.process_window(resampled_134)
                t_prep_end = time.perf_counter()
                preproc_ms = (t_prep_end - t_prep_start) * 1000.0

                # --- Stage 4 & 5: Model Inference & Runtime Wrapper ---
                t_infer_start = time.perf_counter()
                raw_pred = runtime.predict(feat_263)
                t_infer_end = time.perf_counter()

                runtime_wrapper_ms = (t_infer_end - t_infer_start) * 1000.0
                model_only_invoke_ms = float(raw_pred["latency_ms"])
                pred_class = raw_pred["predicted_class"]
                pred_conf = float(raw_pred["confidence"])

            # --- Stage 6: Temporal Prediction Stabilization ---
            t_stab_start = time.perf_counter()
            stab_class, stab_id, stab_conf = stabilizer.update(raw_pred, has_hands)
            t_stab_end = time.perf_counter()
            stabilizer_ms = (t_stab_end - t_stab_start) * 1000.0

            # --- Stage 7: Total Latencies ---
            t_loop_end = time.perf_counter()
            total_e2e_ms = (t_loop_end - t_loop_start) * 1000.0
            pipeline_compute_ms = mediapipe_ms + preproc_ms + runtime_wrapper_ms + stabilizer_ms

            # Budget and contention tracking
            if pipeline_compute_ms > frame_budget_ms:
                over_budget_count += 1
                consecutive_over_budget += 1
                if consecutive_over_budget > max_consecutive_over_budget:
                    max_consecutive_over_budget = consecutive_over_budget
            else:
                consecutive_over_budget = 0

            # Effective frame rates
            compute_fps = 1000.0 / pipeline_compute_ms if pipeline_compute_ms > 0 else 0.0
            loop_fps = 1000.0 / total_e2e_ms if total_e2e_ms > 0 else 0.0

            # Status description
            if telemetry["both_detected"]:
                status = "TRACKING (2 HANDS)"
            elif telemetry["left_detected"]:
                status = "TRACKING (LEFT)"
            elif telemetry["right_detected"]:
                status = "TRACKING (RIGHT)"
            elif buf_len < 64:
                status = f"BUFFERING ({buf_len}/64)"
            else:
                status = "IDLE (NO HANDS)"

            records.append({
                "frame": frame_idx,
                "capture_wait_ms": round(capture_ms, 3),
                "capture_interval_ms": round(capture_interval_ms, 3),
                "mediapipe_ms": round(mediapipe_ms, 3),
                "preproc_ms": round(preproc_ms, 3),
                "model_only_invoke_ms": round(model_only_invoke_ms, 3),
                "runtime_wrapper_ms": round(runtime_wrapper_ms, 3),
                "stabilizer_ms": round(stabilizer_ms, 3),
                "pipeline_compute_ms": round(pipeline_compute_ms, 3),
                "total_e2e_ms": round(total_e2e_ms, 3),
                "compute_fps": round(compute_fps, 2),
                "loop_fps": round(loop_fps, 2),
                "status": status,
                "active_hands": int(telemetry["left_detected"]) + int(telemetry["right_detected"]),
                "predicted_sign": pred_class,
                "confidence": round(pred_conf, 4),
                "stabilized_sign": stab_class,
                "stabilized_conf": round(stab_conf, 4),
                "over_budget": bool(pipeline_compute_ms > frame_budget_ms)
            })

            if frame_idx % 50 == 0 or frame_idx == target_frames:
                print(
                    f"Frame {frame_idx:3d}/{target_frames} | "
                    f"CapWait: {capture_ms:5.1f}ms | "
                    f"MediaPipe: {mediapipe_ms:5.2f}ms | "
                    f"Preproc: {preproc_ms:4.2f}ms | "
                    f"INT8 Model: {model_only_invoke_ms:4.2f}ms | "
                    f"Compute: {pipeline_compute_ms:5.2f}ms | "
                    f"E2E: {total_e2e_ms:5.2f}ms ({loop_fps:4.1f} FPS)"
                )

    finally:
        cap.release()
        extractor.close()
        t_bench_total = time.perf_counter() - t_bench_start

    print("\n[4/4] Processing complete. Computing statistical distributions...")

    df = pd.DataFrame(records)

    # Calculate actual capture FPS across entire profiling run
    actual_captured_frames = len(df)
    elapsed_wall_sec = t_bench_total
    actual_capture_fps = actual_captured_frames / elapsed_wall_sec if elapsed_wall_sec > 0 else 0.0

    # Calculate dropped frame estimate:
    # On a live camera running at target_fps, expected frames = elapsed_wall_sec * target_fps
    expected_frames = elapsed_wall_sec * target_fps
    dropped_frames_est = max(0, int(round(expected_frames - actual_captured_frames)))

    # Compute distribution summaries for each stage
    stages = [
        ("Camera Capture Wait", "capture_wait_ms"),
        ("Camera Frame Interval", "capture_interval_ms"),
        ("MediaPipe Landmark Extraction", "mediapipe_ms"),
        ("Temporal Preprocessing (263-D)", "preproc_ms"),
        ("INT8 TFLite Model Invoke (Pure)", "model_only_invoke_ms"),
        ("EchoGestureRuntime Wrapper", "runtime_wrapper_ms"),
        ("Temporal Stabilization", "stabilizer_ms"),
        ("Pipeline Compute (No Wait)", "pipeline_compute_ms"),
        ("Total End-to-End Loop", "total_e2e_ms")
    ]

    distribution_summaries: Dict[str, Dict[str, float]] = {}
    for stage_label, col_name in stages:
        distribution_summaries[col_name] = compute_distribution_metrics(df[col_name])

    # Save CSV report if path requested
    csv_path = output_csv or str(REPO_ROOT / "reports" / "phase6e_camera_latency.csv")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Per-frame latency report saved to: {csv_path}")

    # Build summary dict
    summary_report: Dict[str, Any] = {
        "metadata": {
            "source": str(source),
            "backend": backend_name,
            "resolution": f"{actual_w}x{actual_h}",
            "target_fps": target_fps,
            "budget_ms_per_frame": round(frame_budget_ms, 2),
            "frames_profiled": actual_captured_frames,
            "warmup_frames": warmup_frames,
            "elapsed_wall_seconds": round(elapsed_wall_sec, 3),
            "actual_capture_fps": round(actual_capture_fps, 2),
            "estimated_dropped_frames": dropped_frames_est,
            "frames_over_budget": int(over_budget_count),
            "over_budget_rate_pct": round((over_budget_count / actual_captured_frames) * 100.0, 2) if actual_captured_frames else 0.0,
            "max_consecutive_over_budget": int(max_consecutive_over_budget),
            "active_inference_mode": active_inference,
            "tflite_num_threads": num_threads
        },
        "distributions": distribution_summaries
    }

    if output_summary:
        Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
        with open(output_summary, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
        print(f"[SAVED] Summary metrics saved to: {output_summary}")

    # Print Formatted Report Table to Console
    print("\n" + "=" * 92)
    print(f"{'ECHOGESTURE PHASE 6E LATENCY DISTRIBUTION REPORT':^92}")
    print(f"{'Source: ' + str(source) + ' | ' + str(actual_captured_frames) + ' Frames | Single-Thread INT8':^92}")
    print("=" * 92)
    print(f"{'Pipeline Stage':<32} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'P99':>8} | {'Max':>8} | {'Unit'}")
    print("-" * 92)

    for stage_label, col_name in stages:
        d = distribution_summaries[col_name]
        print(
            f"{stage_label:<32} | "
            f"{d['mean']:8.3f} | "
            f"{d['median']:8.3f} | "
            f"{d['p95']:8.3f} | "
            f"{d['p99']:8.3f} | "
            f"{d['max']:8.3f} | ms"
        )

    print("-" * 92)
    mean_compute = distribution_summaries["pipeline_compute_ms"]["mean"]
    max_compute_fps = 1000.0 / mean_compute if mean_compute > 0 else 0.0
    mean_loop = distribution_summaries["total_e2e_ms"]["mean"]
    actual_loop_fps = 1000.0 / mean_loop if mean_loop > 0 else 0.0

    print(f"{'Effective Compute Throughput':<32} | {max_compute_fps:8.2f} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8} | FPS")
    print(f"{'Actual Pipeline Operating Rate':<32} | {actual_loop_fps:8.2f} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8} | FPS")
    print("=" * 92)
    print(f"Summary Diagnostics:")
    print(f"  • Frame Budget (at {target_fps} FPS):       {frame_budget_ms:.2f} ms")
    print(f"  • Frames Exceeding Budget:        {over_budget_count}/{actual_captured_frames} ({summary_report['metadata']['over_budget_rate_pct']}%)")
    print(f"  • Max Consecutive Over-Budget:    {max_consecutive_over_budget} frames")
    print(f"  • Actual Capture Stream Rate:     {actual_capture_fps:.2f} FPS")
    print(f"  • Estimated Dropped Frames:       {dropped_frames_est} frames")
    print(f"  • Compute Headroom:               {((frame_budget_ms - mean_compute) / frame_budget_ms) * 100.0:.1f}% remaining")
    print("=" * 92 + "\n")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 6E Runtime Latency Profiler")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or video file path (default: 0)")
    parser.add_argument("--frames", type=int, default=350, help="Number of frames to profile (default: 350)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames to discard (default: 20)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--threads", type=int, default=1, help="TFLite CPU threads (default: 1)")
    parser.add_argument("--conditional-inference", action="store_true",
                        help="Only run model inference when hands are detected (default: False, profiles peak active pipeline)")
    parser.add_argument("--output_csv", type=str, default="reports/phase6e_camera_latency.csv",
                        help="Output path for latency CSV report")
    parser.add_argument("--output_summary", type=str, default="reports/phase6e_camera_latency_summary.json",
                        help="Output path for summary metrics JSON")

    args = parser.parse_args()

    try:
        source_val: Union[int, str] = int(args.source)
    except ValueError:
        source_val = args.source

    profile_pipeline(
        source=source_val,
        target_frames=args.frames,
        warmup_frames=args.warmup,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        active_inference=not args.conditional_inference,
        num_threads=args.threads,
        output_csv=args.output_csv,
        output_summary=args.output_summary
    )
