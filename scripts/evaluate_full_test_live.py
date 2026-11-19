"""
EchoGesture Phase 6D: Full Held-Out Test Set Live Validation
Evaluates the complete 197-clip MS-ASL held-out test partition across:
1. Canonical Offline Preprocessing Pipeline (Whole-Clip Ground Truth)
2. Live Raw Streaming Pipeline (Unflipped, Peak Active Prediction)
3. Live Stabilized Pipeline (Unflipped, 2-Frame Consensus Hysteresis & Decay)

Generates:
- reports/phase6d_full_test_predictions.csv
- reports/phase6d_per_class_metrics.csv
- reports/phase6d_per_signer_metrics.csv
- reports/phase6d_summary_metrics.json
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from scripts.live_inference import (
    LiveLandmarkExtractor,
    LiveTemporalPreprocessor,
    PredictionStabilizer
)
from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.prepare_sequences import resample_sequence


def evaluate_full_test_set(
    manifest_path: str = "data/processed/landmarks_v3/feature_manifest.csv",
    scaler_path: str = "data/processed/sequences_v3/scaler_params.npz",
    model_path: str = "models/quantized/echogesture_exp_d_int8.tflite",
    output_predictions_csv: str = "reports/phase6d_full_test_predictions.csv",
    output_per_class_csv: str = "reports/phase6d_per_class_metrics.csv",
    output_per_signer_csv: str = "reports/phase6d_per_signer_metrics.csv",
    output_summary_json: str = "reports/phase6d_summary_metrics.json",
    limit: Optional[int] = None
):
    print("=================================================================")
    print("EchoGesture Phase 6D: Full Held-Out Test Set Live Validation")
    print("=================================================================")
    print(f"Manifest Path:      {manifest_path}")
    print(f"Scaler Parameters:  {scaler_path}")
    print(f"INT8 TFLite Model:  {model_path}")
    print(f"Output Predictions: {output_predictions_csv}")
    if limit is not None:
        print(f"Limit:              {limit} clips (Dry Run)")
    
    # 1. Load manifest and filter test partition
    df_manifest = pd.read_csv(manifest_path)
    test_df = df_manifest[df_manifest["unified_split"] == "test"].copy()
    test_df.sort_values(by="video_id", inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    if limit is not None and limit > 0:
        test_df = test_df.head(limit).copy()
    
    total_clips = len(test_df)
    print(f"\nLoaded {total_clips} held-out test clips across {test_df['normalized_label'].nunique()} classes and {test_df['signer_id'].nunique()} signers.")
    
    # 2. Initialize Runtime and Pipeline Components
    print("\nInitializing production runtime and pipeline components...")
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path)
    stabilizer = PredictionStabilizer(window_size=5, confidence_threshold=0.35)
    
    idx2label = runtime.idx2label
    label2idx = {v: k for k, v in idx2label.items()}
    all_classes = [idx2label[i] for i in range(len(idx2label))]
    
    prediction_records = []
    failed_clips = []
    
    t_start = time.perf_counter()
    
    # 3. Process every clip in the test set
    for idx, row in test_df.iterrows():
        v_id = row["video_id"]
        v_path = row["video_path"]
        g_truth = row["normalized_label"]
        signer = row["signer_id"]
        total_frames = int(row["total_frames"])
        fps = float(row.get("fps", 30.0))
        duration_sec = float(row.get("duration_sec", total_frames / fps if fps > 0 else 0.0))
        is_short = (total_frames < 64)
        
        # Verify video file existence
        if not os.path.exists(v_path):
            print(f"[{idx+1}/{total_clips}] [ERROR] Missing video file: {v_path}")
            failed_clips.append({"video_id": v_id, "error": "MISSING_VIDEO_FILE"})
            continue
            
        # -------------------------------------------------------------
        # Path A: Canonical Offline Reference (Whole-Clip Preprocessed)
        # -------------------------------------------------------------
        npz_path = row["npz_path"]
        if not os.path.exists(npz_path):
            npz_path = str(REPO_ROOT / "data" / "processed" / "landmarks_v3" / f"{v_id}.npz")
            
        if not os.path.exists(npz_path):
            print(f"[{idx+1}/{total_clips}] [ERROR] Missing offline NPZ: {npz_path}")
            failed_clips.append({"video_id": v_id, "error": "MISSING_OFFLINE_NPZ"})
            continue
            
        off_data = np.load(npz_path)
        off_134 = off_data["flat_features"]
        off_res64 = resample_sequence(off_134, target_len=64)
        off_263 = preprocessor.process_window(off_res64)
        off_pred = runtime.predict(off_263)
        
        off_class = off_pred["predicted_class"]
        off_id = off_pred["predicted_class_id"]
        off_conf = off_pred["confidence"]
        off_top5_classes = [t["class_name"] for t in off_pred["top_5"]]
        off_top5_confs = [t["confidence"] for t in off_pred["top_5"]]
        off_correct = (off_class == g_truth)
        off_top5_correct = (g_truth in off_top5_classes)
        
        # -------------------------------------------------------------
        # Path B & C: Live Streaming on Canonical Un-Flipped Video
        # -------------------------------------------------------------
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[{idx+1}/{total_clips}] [ERROR] Cannot open video: {v_path}")
            failed_clips.append({"video_id": v_id, "error": "CANNOT_OPEN_VIDEO"})
            continue
            
        extractor.reset_tracking()
        stabilizer.reset()
        buf = deque(maxlen=64)
        
        raw_peak_class = "IDLE"
        raw_peak_id = -1
        raw_peak_conf = 0.0
        raw_peak_frame = -1
        raw_peak_top5 = []
        raw_peak_top5_confs = []
        raw_predictions_count = 0
        
        first_stable_frame = -1
        first_stable_sec = -1.0
        stable_changes = 0
        prev_stab = "IDLE"
        prev_stab_id = -1
        prev_stab_conf = 0.0
        
        frame_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_counter += 1
            
            # 1. Feature extraction strictly on canonical un-flipped frame
            f134, telem, _ = extractor.process_frame(frame)
            buf.append(f134)
            buf_len = len(buf)
            has_hands = telem["left_detected"] or telem["right_detected"]
            
            # 2. Streaming Inference
            raw_pred = None
            if buf_len >= 16:
                w_arr = np.array(buf, dtype=np.float32)
                active = np.sum((w_arr[:, 126] > 0.5) | (w_arr[:, 127] > 0.5))
                if active >= 12:
                    res_134 = resample_sequence(w_arr, 64) if buf_len < 64 else w_arr
                    f263 = preprocessor.process_window(res_134)
                    raw_pred = runtime.predict(f263)
                    raw_predictions_count += 1
                    
                    if raw_pred["confidence"] > raw_peak_conf:
                        raw_peak_conf = raw_pred["confidence"]
                        raw_peak_class = raw_pred["predicted_class"]
                        raw_peak_id = raw_pred["predicted_class_id"]
                        raw_peak_frame = frame_counter
                        raw_peak_top5 = [t["class_name"] for t in raw_pred["top_5"]]
                        raw_peak_top5_confs = [t["confidence"] for t in raw_pred["top_5"]]
                        
            # 3. Temporal Stabilization
            stab_c, stab_id, stab_conf = stabilizer.update(raw_pred, has_hands)
            
            if stab_c != "IDLE":
                if first_stable_frame == -1:
                    first_stable_frame = frame_counter
                    first_stable_sec = round(frame_counter / fps, 3) if fps > 0 else 0.0
                if stab_c != prev_stab and prev_stab != "IDLE":
                    stable_changes += 1
                prev_stab = stab_c
                prev_stab_id = stab_id
                prev_stab_conf = stab_conf
                
        cap.release()
        
        # -------------------------------------------------------------
        # Evaluate Outcomes
        # -------------------------------------------------------------
        raw_peak_correct = (raw_peak_class == g_truth)
        raw_peak_top5_correct = (g_truth in raw_peak_top5)
        
        # For stabilized class, take the final non-idle established sign
        stab_final_class = prev_stab
        stab_final_id = prev_stab_id
        stab_final_conf = prev_stab_conf
        stab_final_correct = (stab_final_class == g_truth)
        
        prediction_produced = (raw_predictions_count > 0 and raw_peak_class != "IDLE")
        
        offline_vs_raw_agree = (off_class == raw_peak_class)
        offline_vs_stab_agree = (off_class == stab_final_class)
        
        record = {
            "video_id": v_id,
            "video_path": v_path,
            "ground_truth": g_truth,
            "ground_truth_id": label2idx.get(g_truth, -1),
            "signer_id": signer,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "fps": round(fps, 1),
            "is_short_clip": is_short,
            # Offline metrics
            "offline_pred_class": off_class,
            "offline_pred_id": off_id,
            "offline_conf": round(off_conf, 4),
            "offline_top5": "|".join(off_top5_classes),
            "offline_top5_confs": "|".join([f"{c:.3f}" for c in off_top5_confs]),
            "offline_correct": off_correct,
            "offline_top5_correct": off_top5_correct,
            # Live Raw metrics
            "live_raw_peak_pred": raw_peak_class,
            "live_raw_peak_id": raw_peak_id,
            "live_raw_peak_conf": round(raw_peak_conf, 4),
            "live_raw_peak_frame": raw_peak_frame,
            "live_raw_peak_top5": "|".join(raw_peak_top5),
            "live_raw_peak_top5_confs": "|".join([f"{c:.3f}" for c in raw_peak_top5_confs]),
            "live_raw_correct": raw_peak_correct,
            "live_raw_top5_correct": raw_peak_top5_correct,
            # Live Stabilized metrics
            "live_stab_final_pred": stab_final_class,
            "live_stab_final_id": stab_final_id,
            "live_stab_final_conf": round(stab_final_conf, 4),
            "live_stab_first_frame": first_stable_frame,
            "live_stab_first_sec": first_stable_sec,
            "live_stab_changes": stable_changes,
            "live_stab_correct": stab_final_correct,
            # Parity and reliability
            "prediction_produced": prediction_produced,
            "raw_predictions_count": raw_predictions_count,
            "offline_vs_raw_agree": offline_vs_raw_agree,
            "offline_vs_stab_agree": offline_vs_stab_agree,
            "status": "SUCCESS"
        }
        prediction_records.append(record)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == total_clips:
            elapsed = time.perf_counter() - t_start
            off_acc_so_far = sum(r["offline_correct"] for r in prediction_records) / len(prediction_records)
            raw_acc_so_far = sum(r["live_raw_correct"] for r in prediction_records) / len(prediction_records)
            stab_acc_so_far = sum(r["live_stab_correct"] for r in prediction_records) / len(prediction_records)
            print(f"[{idx+1}/{total_clips}] ({elapsed:.1f}s) | Offline Acc: {off_acc_so_far*100:.1f}% | Raw Acc: {raw_acc_so_far*100:.1f}% | Stab Acc: {stab_acc_so_far*100:.1f}%")
            
    extractor.close()
    
    # 4. Save Prediction CSV
    df_pred = pd.DataFrame(prediction_records)
    Path(output_predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_predictions_csv, index=False)
    print(f"\nSaved full predictions to: {output_predictions_csv}")
    
    # 5. Compute Stage 3 Metrics
    total_evaluated = len(df_pred)
    short_df = df_pred[df_pred["is_short_clip"]]
    long_df = df_pred[~df_pred["is_short_clip"]]
    
    # Overall Accuracies
    offline_top1 = float(accuracy_score(df_pred["ground_truth"], df_pred["offline_pred_class"]))
    offline_top5 = float(df_pred["offline_top5_correct"].mean())
    
    live_raw_top1 = float(accuracy_score(df_pred["ground_truth"], df_pred["live_raw_peak_pred"]))
    live_raw_top5 = float(df_pred["live_raw_top5_correct"].mean())
    
    live_stab_top1 = float(accuracy_score(df_pred["ground_truth"], df_pred["live_stab_final_pred"]))
    
    # F1 scores
    labels_eval = sorted(list(set(df_pred["ground_truth"])))
    
    # Offline F1
    prec_off, rec_off, f1_macro_off, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["offline_pred_class"], labels=labels_eval, average="macro", zero_division=0
    )
    _, _, f1_weighted_off, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["offline_pred_class"], labels=labels_eval, average="weighted", zero_division=0
    )
    
    # Live Raw F1
    _, _, f1_macro_raw, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["live_raw_peak_pred"], labels=labels_eval, average="macro", zero_division=0
    )
    _, _, f1_weighted_raw, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["live_raw_peak_pred"], labels=labels_eval, average="weighted", zero_division=0
    )
    
    # Live Stabilized F1
    _, _, f1_macro_stab, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["live_stab_final_pred"], labels=labels_eval, average="macro", zero_division=0
    )
    _, _, f1_weighted_stab, _ = precision_recall_fscore_support(
        df_pred["ground_truth"], df_pred["live_stab_final_pred"], labels=labels_eval, average="weighted", zero_division=0
    )
    
    # Coverage & Missing
    coverage = float(df_pred["prediction_produced"].mean())
    missing_rate = 1.0 - coverage
    
    # Short vs Long
    short_off_acc = float(short_df["offline_correct"].mean()) if len(short_df) > 0 else 0.0
    short_raw_acc = float(short_df["live_raw_correct"].mean()) if len(short_df) > 0 else 0.0
    short_stab_acc = float(short_df["live_stab_correct"].mean()) if len(short_df) > 0 else 0.0
    
    long_off_acc = float(long_df["offline_correct"].mean()) if len(long_df) > 0 else 0.0
    long_raw_acc = float(long_df["live_raw_correct"].mean()) if len(long_df) > 0 else 0.0
    long_stab_acc = float(long_df["live_stab_correct"].mean()) if len(long_df) > 0 else 0.0
    
    # Prediction Agreement
    agree_raw = float(df_pred["offline_vs_raw_agree"].mean())
    agree_stab = float(df_pred["offline_vs_stab_agree"].mean())
    
    # Latency to First Stable Prediction (for clips that stabilized)
    valid_latencies_frames = df_pred[df_pred["live_stab_first_frame"] > 0]["live_stab_first_frame"]
    valid_latencies_sec = df_pred[df_pred["live_stab_first_sec"] > 0]["live_stab_first_sec"]
    
    median_frames = float(valid_latencies_frames.median()) if len(valid_latencies_frames) > 0 else 0.0
    p95_frames = float(valid_latencies_frames.quantile(0.95)) if len(valid_latencies_frames) > 0 else 0.0
    median_sec = float(valid_latencies_sec.median()) if len(valid_latencies_sec) > 0 else 0.0
    p95_sec = float(valid_latencies_sec.quantile(0.95)) if len(valid_latencies_sec) > 0 else 0.0
    
    # Mean Prediction Changes
    mean_changes = float(df_pred["live_stab_changes"].mean())
    
    # 6. Per-Class Metrics
    per_class_records = []
    for cls_name in sorted(all_classes):
        c_sub = df_pred[df_pred["ground_truth"] == cls_name]
        n_c = len(c_sub)
        if n_c == 0:
            continue
        c_off_acc = float(c_sub["offline_correct"].mean())
        c_raw_acc = float(c_sub["live_raw_correct"].mean())
        c_stab_acc = float(c_sub["live_stab_correct"].mean())
        c_off_top5 = float(c_sub["offline_top5_correct"].mean())
        c_raw_top5 = float(c_sub["live_raw_top5_correct"].mean())
        c_agree = float(c_sub["offline_vs_stab_agree"].mean())
        
        # Recall & Precision for stabilized class
        tp = sum((df_pred["ground_truth"] == cls_name) & (df_pred["live_stab_final_pred"] == cls_name))
        fp = sum((df_pred["ground_truth"] != cls_name) & (df_pred["live_stab_final_pred"] == cls_name))
        fn = sum((df_pred["ground_truth"] == cls_name) & (df_pred["live_stab_final_pred"] != cls_name))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_class_records.append({
            "class_name": cls_name,
            "sample_count": n_c,
            "offline_top1": round(c_off_acc, 4),
            "offline_top5": round(c_off_top5, 4),
            "live_raw_top1": round(c_raw_acc, 4),
            "live_raw_top5": round(c_raw_top5, 4),
            "live_stab_top1": round(c_stab_acc, 4),
            "precision_stab": round(prec, 4),
            "recall_stab": round(rec, 4),
            "f1_stab": round(f1, 4),
            "offline_vs_stab_agreement": round(c_agree, 4)
        })
    df_per_class = pd.DataFrame(per_class_records)
    df_per_class.to_csv(output_per_class_csv, index=False)
    print(f"Saved per-class metrics to: {output_per_class_csv}")
    
    # 7. Per-Signer Metrics
    per_signer_records = []
    for s_name in sorted(list(test_df["signer_id"].unique())):
        s_sub = df_pred[df_pred["signer_id"] == s_name]
        n_s = len(s_sub)
        if n_s == 0:
            continue
        s_off_acc = float(s_sub["offline_correct"].mean())
        s_raw_acc = float(s_sub["live_raw_correct"].mean())
        s_stab_acc = float(s_sub["live_stab_correct"].mean())
        s_agree = float(s_sub["offline_vs_stab_agree"].mean())
        s_changes = float(s_sub["live_stab_changes"].mean())
        
        per_signer_records.append({
            "signer_id": s_name,
            "sample_count": n_s,
            "offline_top1": round(s_off_acc, 4),
            "live_raw_top1": round(s_raw_acc, 4),
            "live_stab_top1": round(s_stab_acc, 4),
            "offline_vs_stab_agreement": round(s_agree, 4),
            "mean_stabilized_changes": round(s_changes, 2)
        })
    df_per_signer = pd.DataFrame(per_signer_records)
    df_per_signer.to_csv(output_per_signer_csv, index=False)
    print(f"Saved per-signer metrics to: {output_per_signer_csv}")
    
    # 8. Summary JSON
    summary = {
        "total_test_clips": total_evaluated,
        "failed_clips_count": len(failed_clips),
        "failed_clips": failed_clips,
        "classes_count": len(all_classes),
        "signers_count": len(per_signer_records),
        "short_clips_count": len(short_df),
        "long_clips_count": len(long_df),
        "metrics": {
            "offline": {
                "top1_accuracy": round(offline_top1, 4),
                "top5_accuracy": round(offline_top5, 4),
                "macro_f1": round(f1_macro_off, 4),
                "weighted_f1": round(f1_weighted_off, 4),
                "short_clip_acc": round(short_off_acc, 4),
                "long_clip_acc": round(long_off_acc, 4)
            },
            "live_raw": {
                "top1_accuracy": round(live_raw_top1, 4),
                "top5_accuracy": round(live_raw_top5, 4),
                "macro_f1": round(f1_macro_raw, 4),
                "weighted_f1": round(f1_weighted_raw, 4),
                "short_clip_acc": round(short_raw_acc, 4),
                "long_clip_acc": round(long_raw_acc, 4)
            },
            "live_stabilized": {
                "top1_accuracy": round(live_stab_top1, 4),
                "macro_f1": round(f1_macro_stab, 4),
                "weighted_f1": round(f1_weighted_stab, 4),
                "short_clip_acc": round(short_stab_acc, 4),
                "long_clip_acc": round(long_stab_acc, 4)
            },
            "coverage": {
                "prediction_coverage": round(coverage, 4),
                "missing_prediction_rate": round(missing_rate, 4)
            },
            "parity_and_stability": {
                "offline_vs_raw_agreement": round(agree_raw, 4),
                "offline_vs_stab_agreement": round(agree_stab, 4),
                "median_time_to_first_prediction_frames": round(median_frames, 1),
                "p95_time_to_first_prediction_frames": round(p95_frames, 1),
                "median_time_to_first_prediction_sec": round(median_sec, 2),
                "p95_time_to_first_prediction_sec": round(p95_sec, 2),
                "mean_prediction_changes": round(mean_changes, 2)
            }
        }
    }
    
    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {output_summary_json}")
    
    print("\n=================================================================")
    print("PHASE 6D EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Total Clips Evaluated: {total_evaluated}")
    print(f"Offline Top-1:         {offline_top1*100:.2f}% (Top-5: {offline_top5*100:.2f}%)")
    print(f"Live Raw Top-1:        {live_raw_top1*100:.2f}% (Top-5: {live_raw_top5*100:.2f}%)")
    print(f"Live Stabilized Top-1: {live_stab_top1*100:.2f}%")
    print(f"Offline/Live Agree:    {agree_stab*100:.2f}%")
    print(f"Missing Clip Rate:     {missing_rate*100:.2f}%")
    print(f"Median Time to Stable: {median_sec:.2f}s ({median_frames:.0f} frames)")
    print("=================================================================")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 6D: Full Test Set Live Validation")
    parser.add_argument("--manifest", type=str, default="data/processed/landmarks_v3/feature_manifest.csv")
    parser.add_argument("--scaler", type=str, default="data/processed/sequences_v3/scaler_params.npz")
    parser.add_argument("--model", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--out-pred", type=str, default="reports/phase6d_full_test_predictions.csv")
    parser.add_argument("--out-class", type=str, default="reports/phase6d_per_class_metrics.csv")
    parser.add_argument("--out-signer", type=str, default="reports/phase6d_per_signer_metrics.csv")
    parser.add_argument("--out-summary", type=str, default="reports/phase6d_summary_metrics.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips for testing")
    args = parser.parse_args()
    
    evaluate_full_test_set(
        manifest_path=args.manifest,
        scaler_path=args.scaler,
        model_path=args.model,
        output_predictions_csv=args.out_pred,
        output_per_class_csv=args.out_class,
        output_per_signer_csv=args.out_signer,
        output_summary_json=args.out_summary,
        limit=args.limit
    )
