"""
EchoGesture Phase 6C: Offline-versus-Live Prediction Parity Audit
Audits feature-level and prediction-level parity between:
1. Canonical Offline Preprocessing Pipeline
2. Live Raw Streaming Pipeline (Unflipped)
3. Live Stabilized Pipeline
4. Live Mirrored Pipeline (Phase 6B Baseline)
Across the 40 held-out MS-ASL test clips.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.live_inference import (
    LiveLandmarkExtractor,
    LiveTemporalPreprocessor,
    PredictionStabilizer
)
from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.prepare_sequences import resample_sequence, apply_scaler
from scripts.augment_sequences import compute_velocity_features


def audit_feature_parity_on_clips(
    manifest_df: pd.DataFrame,
    scaler_path: str,
    model_path: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Stage 3: Quantify exact numerical differences between offline and live feature transformations.
    """
    print("\n=======================================================")
    print("STAGE 3: Preprocessing & Feature Parity Audit")
    print("=======================================================")
    
    scaler_data = np.load(scaler_path)
    scaler_dict = {"mean": scaler_data["mean"], "std": scaler_data["std"]}
    
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    
    # Load TFLite quantization parameters
    interp = tf.lite.Interpreter(model_path=model_path)
    in_details = interp.get_input_details()[0]
    s_in, z_in = in_details["quantization"]
    
    feature_parity_rows = []
    
    # Audit across a diverse 10-clip sample of the 40 test clips
    audit_sample = manifest_df.head(10)
    
    for _, row in audit_sample.iterrows():
        v_id = row["video_id"]
        v_path = row["video_path"]
        g_truth = row.get("ground_truth", row.get("normalized_label"))
        
        # 1. Load canonical Phase 3 offline landmarks
        offline_npz_path = Path("data/processed/landmarks_v3") / f"{v_id}.npz"
        if not offline_npz_path.exists():
            continue
        offline_data = np.load(offline_npz_path)
        offline_134 = offline_data["flat_features"]  # (T, 134)
        
        # 2. Extract live landmarks from raw video (unflipped)
        cap = cv2.VideoCapture(v_path)
        extractor = LiveLandmarkExtractor()
        live_134_list = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            f134, _, _ = extractor.process_frame(frame)
            live_134_list.append(f134)
        cap.release()
        extractor.close()
        live_134 = np.array(live_134_list, dtype=np.float32)
        
        # Ensure identical frame count
        min_T = min(len(offline_134), len(live_134))
        off_134_sub = offline_134[:min_T]
        live_134_sub = live_134[:min_T]
        
        # A. 134-D Raw Features Comparison
        diff_134 = np.abs(off_134_sub - live_134_sub)
        max_diff_134 = float(np.max(diff_134))
        mean_diff_134 = float(np.mean(diff_134))
        match_rate_134 = float(np.mean(diff_134 < 1e-4) * 100.0)
        
        # B. Resampled 64-Frame Tensors Comparison
        off_res64 = resample_sequence(offline_134, target_len=64)
        live_res64 = resample_sequence(live_134, target_len=64)
        diff_res64 = np.abs(off_res64 - live_res64)
        max_diff_res64 = float(np.max(diff_res64))
        mean_diff_res64 = float(np.mean(diff_res64))
        match_rate_res64 = float(np.mean(diff_res64 < 1e-4) * 100.0)
        
        # C. Standardization & Velocity Features (263-D)
        # Canonical offline: apply_scaler + compute_velocity_features
        off_std134 = apply_scaler(off_res64, scaler_dict)
        off_263 = compute_velocity_features(off_std134)
        
        # Live pipeline: LiveTemporalPreprocessor.process_window
        live_263 = preprocessor.process_window(live_res64)
        
        diff_263 = np.abs(off_263 - live_263)
        max_diff_263 = float(np.max(diff_263))
        mean_diff_263 = float(np.mean(diff_263))
        match_rate_263 = float(np.mean(diff_263 < 1e-3) * 100.0)
        
        # D. Preprocessing Math Parity (Same input passed to both preprocessors)
        live_std_on_off = preprocessor.process_window(off_res64)
        diff_math = np.abs(off_263 - live_std_on_off)
        max_diff_math = float(np.max(diff_math))
        
        # E. INT8 Quantized Input Tensors
        q_off = np.clip(np.round(off_263 / s_in) + z_in, -128, 127).astype(np.int8)
        q_live = np.clip(np.round(live_263 / s_in) + z_in, -128, 127).astype(np.int8)
        diff_q = np.abs(q_off.astype(np.int32) - q_live.astype(np.int32))
        max_diff_q = int(np.max(diff_q))
        exact_match_q = float(np.mean(q_off == q_live) * 100.0)
        
        feature_parity_rows.append({
            "video_id": v_id,
            "ground_truth": g_truth,
            "frames_count": min_T,
            "max_diff_134": round(max_diff_134, 6),
            "mean_diff_134": round(mean_diff_134, 6),
            "match_rate_134_pct": round(match_rate_134, 2),
            "max_diff_resampled_64": round(max_diff_res64, 6),
            "match_rate_resampled_64_pct": round(match_rate_res64, 2),
            "max_diff_math_263": round(max_diff_math, 6),
            "max_diff_final_263": round(max_diff_263, 6),
            "match_rate_final_263_pct": round(match_rate_263, 2),
            "max_diff_int8_tensor": max_diff_q,
            "exact_match_int8_pct": round(exact_match_q, 2)
        })
        print(f"  [{v_id}] 134-D Diff: {max_diff_134:.4f} | Math 263-D Diff: {max_diff_math:.6f} | INT8 Match: {exact_match_q:.1f}%")
        
    df_fp = pd.DataFrame(feature_parity_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_fp.to_csv(output_csv, index=False)
    print(f"Saved feature parity audit to {output_csv}")
    
    print("\n--- STAGE 3 FEATURE PARITY SUMMARY ---")
    print(f"Mean 134-D Match Rate:           {df_fp['match_rate_134_pct'].mean():.2f}%")
    print(f"Mean Resampled 64-D Match Rate:   {df_fp['match_rate_resampled_64_pct'].mean():.2f}%")
    print(f"Max Preprocessor Math Epsilon:    {df_fp['max_diff_math_263'].max():.6e}")
    print(f"Mean INT8 Quantized Match Rate:   {df_fp['exact_match_int8_pct'].mean():.2f}%")
    
    return df_fp


def run_full_parity_experiment(
    manifest_path: str,
    scaler_path: str,
    model_path: str,
    output_clip_csv: str,
    output_feature_csv: str
):
    """
    Stage 2: Run all 4 paths (Offline, Live Unflipped Raw, Live Stabilized, Live Mirrored)
    across the exact 40 Phase 6B test clips.
    """
    print("\n=======================================================")
    print("STAGE 2: Controlled 4-Path Prediction Parity Experiment")
    print("=======================================================")
    
    # Load 40 clips from Phase 6B evaluation CSV for exact consistency
    phase6b_csv = "reports/phase6b_multiclip_evaluation.csv"
    if not os.path.exists(phase6b_csv):
        raise FileNotFoundError(f"Missing Phase 6B CSV at {phase6b_csv}")
    df_40 = pd.read_csv(phase6b_csv)
    
    # 1. Run Feature Parity Audit (Stage 3)
    audit_feature_parity_on_clips(df_40, scaler_path, model_path, output_feature_csv)
    
    # 2. Run Clip Parity Comparison (Stage 2)
    print("\nEvaluating 40 test clips across 4 execution pathways...")
    
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path)
    stabilizer_unflipped = PredictionStabilizer(window_size=5, confidence_threshold=0.35)
    stabilizer_mirrored = PredictionStabilizer(window_size=5, confidence_threshold=0.35)
    
    comparison_records = []
    
    for idx, row in df_40.iterrows():
        v_id = row["video_id"]
        v_path = row["video_path"]
        g_truth = row.get("ground_truth", row.get("normalized_label"))
        signer = row["signer_id"]
        
        # ---------------------------------------------------------
        # Path A: Canonical Offline Prediction (Whole-Clip Preprocessed)
        # ---------------------------------------------------------
        offline_npz_path = Path("data/processed/landmarks_v3") / f"{v_id}.npz"
        offline_data = np.load(offline_npz_path)
        offline_134 = offline_data["flat_features"]
        
        off_res64 = resample_sequence(offline_134, target_len=64)
        off_263 = preprocessor.process_window(off_res64)
        off_pred = runtime.predict(off_263)
        
        off_class = off_pred["predicted_class"]
        off_conf = off_pred["confidence"]
        off_top5 = [t["class_name"] for t in off_pred["top_5"]]
        off_correct = (off_class == g_truth)
        
        # ---------------------------------------------------------
        # Path B & C: Live Streaming on Raw Video (Unflipped)
        # ---------------------------------------------------------
        cap = cv2.VideoCapture(v_path)
        extractor.reset_tracking()
        stabilizer_unflipped.reset()
        buf_unflipped = deque(maxlen=64)
        
        raw_peak_class = "IDLE"
        raw_peak_conf = 0.0
        raw_peak_frame = -1
        raw_top5_at_peak = []
        
        first_stable_frame = -1
        stable_changes = 0
        prev_stab = "IDLE"
        
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_counter += 1
            
            # RAW UNFLIPPED EXTRACTION
            f134, telem, _ = extractor.process_frame(frame)
            buf_unflipped.append(f134)
            buf_len = len(buf_unflipped)
            has_hands = telem["left_detected"] or telem["right_detected"]
            
            raw_pred = None
            if buf_len >= 16:
                w_arr = np.array(buf_unflipped, dtype=np.float32)
                active = np.sum((w_arr[:, 126] > 0.5) | (w_arr[:, 127] > 0.5))
                if active >= 12:
                    res_134 = resample_sequence(w_arr, 64) if buf_len < 64 else w_arr
                    f263 = preprocessor.process_window(res_134)
                    raw_pred = runtime.predict(f263)
                    
                    if raw_pred["confidence"] > raw_peak_conf:
                        raw_peak_conf = raw_pred["confidence"]
                        raw_peak_class = raw_pred["predicted_class"]
                        raw_peak_frame = frame_counter
                        raw_top5_at_peak = [t["class_name"] for t in raw_pred.get("top_5", [])]
                        
            # Stabilized update
            stab_c, stab_id, stab_conf = stabilizer_unflipped.update(raw_pred, has_hands)
            if stab_c != "IDLE" and first_stable_frame == -1:
                first_stable_frame = frame_counter
            if stab_c != prev_stab:
                if prev_stab != "IDLE":
                    stable_changes += 1
                prev_stab = stab_c
                
        cap.release()
        
        raw_peak_correct = (raw_peak_class == g_truth)
        stab_final_class = prev_stab
        stab_final_correct = (stab_final_class == g_truth)
        
        # ---------------------------------------------------------
        # Path D: Live Streaming on Mirrored Video (Phase 6B Baseline)
        # ---------------------------------------------------------
        cap_m = cv2.VideoCapture(v_path)
        extractor.reset_tracking()
        stabilizer_mirrored.reset()
        buf_mirrored = deque(maxlen=64)
        
        prev_m_stab = "IDLE"
        m_peak_class = "IDLE"
        m_peak_conf = 0.0
        
        f_m_idx = 0
        while True:
            ret, frame = cap_m.read()
            if not ret or frame is None:
                break
            f_m_idx += 1
            
            # FLIPPED / MIRRORED FRAME
            mirrored_f = cv2.flip(frame, 1)
            f134_m, telem_m, _ = extractor.process_frame(mirrored_f)
            buf_mirrored.append(f134_m)
            buf_m_len = len(buf_mirrored)
            has_hands_m = telem_m["left_detected"] or telem_m["right_detected"]
            
            raw_m_pred = None
            if buf_m_len >= 16:
                w_m_arr = np.array(buf_mirrored, dtype=np.float32)
                active_m = np.sum((w_m_arr[:, 126] > 0.5) | (w_m_arr[:, 127] > 0.5))
                if active_m >= 12:
                    res_m_134 = resample_sequence(w_m_arr, 64) if buf_m_len < 64 else w_m_arr
                    f263_m = preprocessor.process_window(res_m_134)
                    raw_m_pred = runtime.predict(f263_m)
                    if raw_m_pred["confidence"] > m_peak_conf:
                        m_peak_conf = raw_m_pred["confidence"]
                        m_peak_class = raw_m_pred["predicted_class"]
                        
            stab_m_c, _, _ = stabilizer_mirrored.update(raw_m_pred, has_hands_m)
            if stab_m_c != prev_m_stab:
                prev_m_stab = stab_m_c
                
        cap_m.release()
        mirrored_correct = (prev_m_stab == g_truth) or (m_peak_class == g_truth)
        
        comparison_records.append({
            "video_id": v_id,
            "ground_truth": g_truth,
            "signer_id": signer,
            "total_frames": frame_counter,
            # Path A: Offline
            "offline_pred": off_class,
            "offline_conf": round(off_conf, 3),
            "offline_top5": "|".join(off_top5),
            "offline_correct": off_correct,
            # Path B: Live Raw Unflipped
            "live_raw_peak_pred": raw_peak_class,
            "live_raw_peak_conf": round(raw_peak_conf, 3),
            "live_raw_peak_frame": raw_peak_frame,
            "live_raw_correct": raw_peak_correct,
            # Path C: Live Stabilized Unflipped
            "live_stab_final_pred": stab_final_class,
            "live_stab_first_frame": first_stable_frame,
            "live_stab_changes": stable_changes,
            "live_stab_correct": stab_final_correct,
            # Path D: Live Mirrored (Phase 6B)
            "mirrored_peak_pred": m_peak_class,
            "mirrored_peak_conf": round(m_peak_conf, 3),
            "mirrored_final_pred": prev_m_stab,
            "mirrored_correct": mirrored_correct,
            # Agreement flags
            "offline_vs_raw_agree": (off_class == raw_peak_class),
            "offline_vs_stab_agree": (off_class == stab_final_class)
        })
        
        status_str = f"GT: {g_truth:12s} | Off: {off_class:12s} ({off_conf:.2f}) | LiveRaw: {raw_peak_class:12s} ({raw_peak_conf:.2f}) | Mirrored: {m_peak_class:12s}"
        print(f"[{idx+1:02d}/40] {v_id}: {status_str}")
        
    extractor.close()
    
    df_comp = pd.DataFrame(comparison_records)
    Path(output_clip_csv).parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(output_clip_csv, index=False)
    print(f"\nSaved clip parity comparison to {output_clip_csv}")
    
    print("\n=======================================================")
    print("PARITY EXPERIMENT SUMMARY (40 HELD-OUT CLIPS)")
    print("=======================================================")
    acc_off = df_comp["offline_correct"].mean() * 100.0
    acc_raw = df_comp["live_raw_correct"].mean() * 100.0
    acc_stab = df_comp["live_stab_correct"].mean() * 100.0
    acc_mirr = df_comp["mirrored_correct"].mean() * 100.0
    agree_off_raw = df_comp["offline_vs_raw_agree"].mean() * 100.0
    
    print(f"1. Canonical Offline Accuracy:       {acc_off:.2f}% ({df_comp['offline_correct'].sum()}/40)")
    print(f"2. Live Raw Unflipped Accuracy:      {acc_raw:.2f}% ({df_comp['live_raw_correct'].sum()}/40)")
    print(f"3. Live Stabilized Unflipped Accuracy:{acc_stab:.2f}% ({df_comp['live_stab_correct'].sum()}/40)")
    print(f"4. Live Mirrored Accuracy (Phase 6B): {acc_mirr:.2f}% ({df_comp['mirrored_correct'].sum()}/40)")
    print(f"\nPrediction Agreement (Offline vs Live Raw): {agree_off_raw:.2f}%")
    
    return df_comp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 6C Parity Audit")
    parser.add_argument("--manifest", type=str, default="data/splits/normalized_multidataset_manifest.csv")
    parser.add_argument("--scaler", type=str, default="data/processed/sequences_v3/scaler_params.npz")
    parser.add_argument("--model", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--clip_csv", type=str, default="reports/phase6c_clip_comparison.csv")
    parser.add_argument("--feature_csv", type=str, default="reports/phase6c_feature_parity.csv")
    args = parser.parse_args()
    
    run_full_parity_experiment(args.manifest, args.scaler, args.model, args.clip_csv, args.feature_csv)
