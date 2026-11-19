"""
EchoGesture Phase 6B: Temporal Reliability and Recognition Robustness Audit
Evaluates the live inference pipeline against held-out MS-ASL test clips under
both native and controlled perturbation conditions.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Callable

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd

from scripts.live_inference import (
    LiveLandmarkExtractor,
    LiveTemporalPreprocessor,
    PredictionStabilizer
)
from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.prepare_sequences import resample_sequence


def select_representative_test_clips(manifest_path: str, seed: int = 42) -> pd.DataFrame:
    """
    Select 40 representative test clips (1 per class) covering diverse signers and durations.
    """
    df = pd.read_csv(manifest_path)
    test_msasl = df[(df["unified_split"] == "test") & (df["dataset_source"] == "MS-ASL")].copy()
    
    # Filter to existing video paths
    test_msasl = test_msasl[test_msasl["video_path"].apply(os.path.exists)].reset_index(drop=True)
    
    selected_rows = []
    # Deterministic sampling per class
    for label, group in test_msasl.groupby("normalized_label"):
        sorted_group = group.sort_values(by="video_id")
        sample = sorted_group.sample(n=1, random_state=seed).iloc[0]
        selected_rows.append(sample)
        
    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected_df


def evaluate_single_clip(
    video_path: str,
    extractor: LiveLandmarkExtractor,
    preprocessor: LiveTemporalPreprocessor,
    runtime: EchoGestureRuntime,
    stabilizer: PredictionStabilizer,
    perturbation_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
    buffer_len: int = 64,
    enable_incomplete_window_resampling: bool = True
) -> Dict[str, Any]:
    """
    Process a single video clip through the live inference pipeline frame-by-frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "ERROR_OPENING_VIDEO", "error_reason": "CANNOT_OPEN_FILE"}
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Reset temporal state for clean independent clip evaluation
    extractor.reset_tracking()
    stabilizer.reset()
    buffer_134 = deque(maxlen=buffer_len)
    
    timeline = []
    frame_idx = 0
    
    first_inference_frame = -1
    first_stable_frame = -1
    first_stable_class = "IDLE"
    first_stable_conf = 0.0
    
    peak_confidence = 0.0
    peak_class = "IDLE"
    top_5_at_peak = []
    
    prediction_changes = 0
    prev_stab_class = "IDLE"
    unique_classes_seen = ["IDLE"]
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_idx += 1
        
        # Apply perturbation if provided
        if perturbation_fn is not None:
            frame = perturbation_fn(frame, frame_idx, total_frames)
            
        # Horizontal flip (mirror) matching live camera ingestion
        mirrored = cv2.flip(frame, 1)
        
        # 1. MediaPipe feature extraction
        flat_134, telemetry, _ = extractor.process_frame(mirrored)
        buffer_134.append(flat_134)
        buf_len = len(buffer_134)
        has_hands = telemetry["left_detected"] or telemetry["right_detected"]
        
        raw_pred = None
        # 2. Incomplete window handling & rolling buffer execution condition
        min_frames = 16 if enable_incomplete_window_resampling else 64
        if buf_len >= min_frames:
            window_arr = np.array(buffer_134, dtype=np.float32)
            active_frames = np.sum((window_arr[:, 126] > 0.5) | (window_arr[:, 127] > 0.5))
            if active_frames >= 12:
                if first_inference_frame == -1:
                    first_inference_frame = frame_idx
                if buf_len < 64:
                    resampled_134 = resample_sequence(window_arr, target_len=64)
                else:
                    resampled_134 = window_arr
                feat_263 = preprocessor.process_window(resampled_134)
                raw_pred = runtime.predict(feat_263)
                
                if raw_pred["confidence"] > peak_confidence:
                    peak_confidence = raw_pred["confidence"]
                    peak_class = raw_pred["predicted_class"]
                    top_5_at_peak = raw_pred.get("top_5", [])
                    
        # 3. Stabilization
        stab_class, stab_id, stab_conf = stabilizer.update(raw_pred, has_hands)
        
        # Track first non-IDLE stable prediction
        if stab_class != "IDLE" and first_stable_frame == -1:
            first_stable_frame = frame_idx
            first_stable_class = stab_class
            first_stable_conf = stab_conf
            
        # Track prediction changes
        if stab_class != prev_stab_class:
            if prev_stab_class != "IDLE":
                prediction_changes += 1
            prev_stab_class = stab_class
            if stab_class not in unique_classes_seen:
                unique_classes_seen.append(stab_class)
                
        timeline.append({
            "frame": frame_idx,
            "has_hands": has_hands,
            "raw_class": raw_pred["predicted_class"] if raw_pred else None,
            "raw_conf": raw_pred["confidence"] if raw_pred else 0.0,
            "stab_class": stab_class,
            "stab_conf": stab_conf
        })
        
    cap.release()
    
    final_pred_class = prev_stab_class
    final_conf = timeline[-1]["stab_conf"] if timeline else 0.0
    
    # Classify outcome
    is_missing = (first_stable_frame == -1)
    
    if is_missing:
        if total_frames < 64:
            error_reason = "SHORT_CLIP_BUFFER_INCOMPLETE"
        else:
            error_reason = "NO_HANDS_OR_LOW_CONFIDENCE"
    else:
        error_reason = "NONE"
        
    return {
        "status": "SUCCESS",
        "clip_length_frames": total_frames,
        "frames_evaluated": frame_idx,
        "fps": fps,
        "first_inference_frame": first_inference_frame,
        "first_stable_frame": first_stable_frame,
        "first_stable_time_sec": round(first_stable_frame / fps, 3) if first_stable_frame != -1 else -1.0,
        "first_stable_class": first_stable_class,
        "first_stable_conf": round(first_stable_conf, 3),
        "final_predicted_class": final_pred_class,
        "final_confidence": round(final_conf, 3),
        "peak_confidence": round(peak_confidence, 3),
        "peak_class": peak_class,
        "top_5_at_peak": json.dumps(top_5_at_peak),
        "prediction_changes": prediction_changes,
        "unique_classes_seen": "|".join(unique_classes_seen),
        "is_missing": is_missing,
        "error_reason": error_reason,
        "timeline_len": len(timeline)
    }


def run_stage2_multiclip_audit(
    manifest_path: str,
    scaler_path: str,
    model_path: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Execute Stage 2 multi-clip evaluation across 40 representative held-out test clips.
    """
    print("\n=======================================================")
    print("STAGE 2: Multi-Clip Held-Out Test Evaluation")
    print("=======================================================")
    
    selected_df = select_representative_test_clips(manifest_path, seed=42)
    print(f"Selected {len(selected_df)} representative MS-ASL test clips across {selected_df['normalized_label'].nunique()} classes and {selected_df['signer_id'].nunique()} signers.")
    
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path)
    stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35, idle_dropout_frames=12)
    
    results = []
    for idx, row in selected_df.iterrows():
        v_id = row["video_id"]
        v_path = row["video_path"]
        g_truth = row["normalized_label"]
        signer = row["signer_id"]
        
        print(f"[{idx+1:02d}/40] Testing {v_id} (Label: {g_truth}, Signer: {signer})...", end=" ", flush=True)
        t0 = time.perf_counter()
        eval_res = evaluate_single_clip(v_path, extractor, preprocessor, runtime, stabilizer)
        elapsed = time.perf_counter() - t0
        
        pred_class = eval_res["final_predicted_class"]
        peak_c = eval_res["peak_class"]
        is_correct = (pred_class == g_truth) or (peak_c == g_truth)
        
        if eval_res["is_missing"]:
            status_tag = f"MISSING ({eval_res['error_reason']})"
        elif is_correct:
            status_tag = f"CORRECT ({peak_c}, conf={eval_res['peak_confidence']:.2f})"
        else:
            status_tag = f"MISCLASSIFIED (Pred: {peak_c}, GT: {g_truth})"
            
        print(f"{status_tag} in {elapsed:.1f}s")
        
        res_record = {
            "video_id": v_id,
            "signer_id": signer,
            "ground_truth": g_truth,
            "video_path": v_path,
            "clip_length_frames": eval_res["clip_length_frames"],
            "first_inference_frame": eval_res["first_inference_frame"],
            "first_stable_frame": eval_res["first_stable_frame"],
            "first_stable_time_sec": eval_res["first_stable_time_sec"],
            "first_stable_class": eval_res["first_stable_class"],
            "first_stable_conf": eval_res["first_stable_conf"],
            "final_predicted_class": eval_res["final_predicted_class"],
            "final_confidence": eval_res["final_confidence"],
            "peak_confidence": eval_res["peak_confidence"],
            "peak_class": eval_res["peak_class"],
            "top_5_at_peak": eval_res["top_5_at_peak"],
            "prediction_changes": eval_res["prediction_changes"],
            "unique_classes_seen": eval_res["unique_classes_seen"],
            "is_correct": is_correct,
            "is_missing": eval_res["is_missing"],
            "error_reason": eval_res["error_reason"]
        }
        results.append(res_record)
        
    extractor.close()
    
    df_results = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nMulti-clip evaluation saved to {output_csv}")
    
    # Summary Analysis
    total = len(df_results)
    short_clips = df_results[df_results["clip_length_frames"] < 64]
    long_clips = df_results[df_results["clip_length_frames"] >= 64]
    
    print("\n--- STAGE 2 AUDIT SUMMARY ---")
    print(f"Total clips evaluated: {total}")
    print(f"Clips < 64 frames:     {len(short_clips)} ({len(short_clips)/total*100:.1f}%)")
    print(f"Clips >= 64 frames:    {len(long_clips)} ({len(long_clips)/total*100:.1f}%)")
    print(f"Missing predictions:   {df_results['is_missing'].sum()} ({df_results['is_missing'].mean()*100:.1f}%)")
    print(f"Missing on short:      {short_clips['is_missing'].sum()} / {len(short_clips)}")
    print(f"Accuracy on long:      {long_clips['is_correct'].mean()*100:.1f}% ({long_clips['is_correct'].sum()}/{len(long_clips)})")
    print(f"Overall Accuracy:      {df_results['is_correct'].mean()*100:.1f}% ({df_results['is_correct'].sum()}/{total})")
    
    return df_results


def run_stage3_robustness_audit(
    manifest_path: str,
    scaler_path: str,
    model_path: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Execute Stage 3 controlled robustness perturbations across representative test clips.
    """
    print("\n=======================================================")
    print("STAGE 3: Controlled Robustness Perturbation Scenarios")
    print("=======================================================")
    
    # Select 6 diverse clips with frame count >= 64 so that baseline inference triggers reliably
    selected_df = select_representative_test_clips(manifest_path, seed=42)
    test_candidates = selected_df[selected_df["video_path"].apply(lambda p: cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT) >= 64)].head(6)
    
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path)
    stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35, idle_dropout_frames=12)
    
    # Define Controlled Perturbation Functions
    def perturb_dimmed_0_4(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        return np.clip(frame.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
        
    def perturb_occlusion_lower_30pct(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        f_occ = frame.copy()
        h, w, _ = f_occ.shape
        f_occ[int(h * 0.7):, :] = 0  # Black out lower 30% of frame (wrist / lower lap area)
        return f_occ
        
    def perturb_tracking_drop_5frames(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        mid = total_f // 2
        if mid <= f_idx < mid + 5:
            return np.zeros_like(frame)  # 5-frame blackout (below 12-frame dropout)
        return frame
        
    def perturb_tracking_drop_15frames(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        mid = total_f // 2
        if mid <= f_idx < mid + 15:
            return np.zeros_like(frame)  # 15-frame blackout (exceeds 12-frame dropout)
        return frame
        
    def perturb_spatial_shift_10pct(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        h, w, _ = frame.shape
        M = np.float32([[1, 0, w * 0.08], [0, 1, h * 0.08]])
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
    def perturb_scale_down_0_75x(frame: np.ndarray, f_idx: int, total_f: int) -> np.ndarray:
        h, w, _ = frame.shape
        scaled = cv2.resize(frame, (int(w * 0.75), int(h * 0.75)))
        canvas = np.zeros_like(frame)
        pad_y = (h - scaled.shape[0]) // 2
        pad_x = (w - scaled.shape[1]) // 2
        canvas[pad_y:pad_y + scaled.shape[0], pad_x:pad_x + scaled.shape[1]] = scaled
        return canvas

    scenarios = [
        ("baseline", None, "Unaltered original video"),
        ("dimmed_0.4", perturb_dimmed_0_4, "60% brightness reduction (severe underexposure)"),
        ("occlusion_lower_30pct", perturb_occlusion_lower_30pct, "Lower 30% frame masked to zero"),
        ("tracking_drop_5frames", perturb_tracking_drop_5frames, "5-frame tracking blackout in mid-gesture"),
        ("tracking_drop_15frames", perturb_tracking_drop_15frames, "15-frame tracking blackout in mid-gesture"),
        ("spatial_shift_10pct", perturb_spatial_shift_10pct, "8% horizontal and vertical translation"),
        ("scale_down_0.75x", perturb_scale_down_0_75x, "25% distance scaling / zoom out")
    ]
    
    robustness_records = []
    
    for _, row in test_candidates.iterrows():
        v_id = row["video_id"]
        v_path = row["video_path"]
        g_truth = row["normalized_label"]
        signer = row["signer_id"]
        
        print(f"\n--- Testing Robustness for {v_id} (GT: {g_truth}) ---")
        for sc_name, sc_fn, sc_desc in scenarios:
            eval_res = evaluate_single_clip(v_path, extractor, preprocessor, runtime, stabilizer, perturbation_fn=sc_fn)
            pred_class = eval_res["final_predicted_class"]
            peak_c = eval_res["peak_class"]
            is_correct = (pred_class == g_truth) or (peak_c == g_truth)
            
            print(f"  [{sc_name:25s}] Peak: {peak_c:12s} ({eval_res['peak_confidence']:.2f}) | Final: {pred_class:12s} | Correct: {is_correct} | Missing: {eval_res['is_missing']}")
            
            robustness_records.append({
                "video_id": v_id,
                "ground_truth": g_truth,
                "signer_id": signer,
                "scenario": sc_name,
                "scenario_description": sc_desc,
                "clip_length_frames": eval_res["clip_length_frames"],
                "first_stable_frame": eval_res["first_stable_frame"],
                "first_stable_time_sec": eval_res["first_stable_time_sec"],
                "peak_class": peak_c,
                "peak_confidence": eval_res["peak_confidence"],
                "final_predicted_class": pred_class,
                "final_confidence": eval_res["final_confidence"],
                "prediction_changes": eval_res["prediction_changes"],
                "is_correct": is_correct,
                "is_missing": eval_res["is_missing"],
                "error_reason": eval_res["error_reason"]
            })
            
    extractor.close()
    
    df_rob = pd.DataFrame(robustness_records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_rob.to_csv(output_csv, index=False)
    print(f"\nRobustness evaluation saved to {output_csv}")
    
    print("\n--- STAGE 3 SCENARIO SUMMARY ---")
    summary = df_rob.groupby("scenario").agg(
        accuracy=("is_correct", "mean"),
        mean_peak_conf=("peak_confidence", "mean"),
        missing_rate=("is_missing", "mean"),
        mean_changes=("prediction_changes", "mean")
    ).reset_index()
    print(summary.to_string(index=False))
    
    return df_rob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 6B Reliability & Robustness Audit")
    parser.add_argument("--manifest", type=str, default="data/splits/normalized_multidataset_manifest.csv")
    parser.add_argument("--scaler", type=str, default="data/processed/sequences_v3/scaler_params.npz")
    parser.add_argument("--model", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--multiclip_csv", type=str, default="reports/phase6b_multiclip_evaluation.csv")
    parser.add_argument("--robustness_csv", type=str, default="reports/phase6b_robustness_evaluation.csv")
    args = parser.parse_args()
    
    run_stage2_multiclip_audit(args.manifest, args.scaler, args.model, args.multiclip_csv)
    run_stage3_robustness_audit(args.manifest, args.scaler, args.model, args.robustness_csv)
