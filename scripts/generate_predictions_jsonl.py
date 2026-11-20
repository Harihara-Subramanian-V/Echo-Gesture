"""
EchoGesture Submission Utility: generate_predictions_jsonl.py
Track 02: On-Device Real-Time ASL Recognition System (Samsung XR Hackathon 2026)

Generates official predictions.jsonl adhering to evaluation specifications.
Supports three ingestion pathways:
1. Pre-extracted feature sequences (.npz) -- e.g. data/processed/sequences_v3/test_sequences.npz
2. Evaluation manifest (.csv)           -- e.g. official evaluation manifest or feature_manifest.csv
3. Raw video directory (.mp4, .avi, etc.) -- runs full MediaPipe landmark extraction & temporal pipeline

Output format (exact JSON Lines, 1 line per clip):
{"clip_id": "...", "prediction": "UPPERCASE_GLOSS", "confidence": 0.xx, "top5": ["GLOSS1", ...], "latency_ms": xx.x}

If an official external evaluation manifest is provided later, this script processes it
directly via `--manifest <path_to_manifest.csv>`.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# Suppress verbose TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.augment_sequences import compute_velocity_features, batch_compute_velocity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EchoGesture predictions.jsonl for hackathon evaluation."
    )
    # Input modes (mutually exclusive or prioritized: sequences > manifest > video_dir)
    parser.add_argument(
        "--sequences",
        type=str,
        default="data/processed/sequences_v3/test_sequences.npz",
        help="Path to pre-extracted sequences .npz file (e.g. data/processed/sequences_v3/test_sequences.npz)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to evaluation manifest CSV containing video paths or IDs."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Path to directory containing raw evaluation videos (.mp4, .avi, etc.)."
    )
    # Output and Model paths
    parser.add_argument(
        "--output",
        type=str,
        default="submission/predictions.jsonl",
        help="Output path for generated predictions.jsonl file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/quantized/echogesture_exp_d_int8.tflite",
        help="Path to frozen INT8 quantized TFLite model."
    )
    parser.add_argument(
        "--label-encoder",
        type=str,
        default="data/processed/sequences_v3/label_encoder.json",
        help="Path to label encoder mapping JSON."
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="data/processed/sequences_v3/scaler_params.npz",
        help="Path to sequence normalizer scaler parameters .npz."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for TFLite interpreter (default: 1)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of clips to process (for dry-run testing)."
    )
    return parser.parse_args()


def process_sequences_npz(
    sequences_path: str,
    runtime: EchoGestureRuntime,
    limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, float]]]:
    """
    Generate predictions from pre-extracted sequences .npz archive.
    """
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")
        
    print(f"Loading pre-extracted sequences from: {sequences_path}")
    data = np.load(sequences_path, allow_pickle=True)
    
    # Identify feature array
    if "X" in data:
        X = data["X"]
    elif "features" in data:
        X = data["features"]
    elif "sequences" in data:
        X = data["sequences"]
    else:
        raise KeyError(f"Could not find sequence feature tensor in {list(data.keys())}")
        
    # Identify video/clip IDs
    if "video_ids" in data:
        clip_ids = [str(vid) for vid in data["video_ids"]]
    elif "clip_ids" in data:
        clip_ids = [str(cid) for cid in data["clip_ids"]]
    elif "ids" in data:
        clip_ids = [str(i) for i in data["ids"]]
    else:
        clip_ids = [f"test_clip_{i:04d}" for i in range(len(X))]
        
    y_true = data.get("y", None)
    
    if limit is not None and limit > 0:
        X = X[:limit]
        clip_ids = clip_ids[:limit]
        if y_true is not None:
            y_true = y_true[:limit]
            
    num_samples = len(X)
    print(f"Loaded {num_samples} sequences with shape {X.shape}.")
    
    # Check feature dimensionality
    if X.ndim == 3 and X.shape[-1] == 134:
        print("Input sequences are 134-D. Computing 263-D velocity features...")
        X_263 = batch_compute_velocity(X)
    elif X.ndim == 3 and X.shape[-1] == 263:
        X_263 = X.astype(np.float32)
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}. Expected (N, 64, 134) or (N, 64, 263).")
        
    predictions = []
    correct_top1 = 0
    correct_top5 = 0
    
    for i in range(num_samples):
        clip_id = clip_ids[i]
        sample = X_263[i]
        
        res = runtime.predict(sample)
        
        pred_gloss = str(res["predicted_class"]).upper()
        conf = round(float(res["confidence"]), 4)
        top5_glosses = [str(item["class_name"]).upper() for item in res["top_5"]]
        lat_ms = round(float(res["latency_ms"]), 2)
        
        record = {
            "clip_id": clip_id,
            "prediction": pred_gloss,
            "confidence": conf,
            "top5": top5_glosses,
            "latency_ms": lat_ms
        }
        predictions.append(record)
        
        if y_true is not None:
            true_idx = int(y_true[i])
            true_label = runtime.idx2label.get(true_idx, "").upper()
            if pred_gloss == true_label:
                correct_top1 += 1
            if true_label in top5_glosses:
                correct_top5 += 1
                
    metrics = None
    if y_true is not None and num_samples > 0:
        metrics = {
            "top1_accuracy": correct_top1 / num_samples,
            "top5_accuracy": correct_top5 / num_samples,
            "top1_count": correct_top1,
            "top5_count": correct_top5,
            "total_samples": num_samples
        }
        
    return predictions, metrics


def process_manifest_csv(
    manifest_path: str,
    scaler_path: str,
    runtime: EchoGestureRuntime,
    limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, float]]]:
    """
    Generate predictions from an evaluation manifest CSV.
    Supports either pre-extracted .npz landmark paths or raw video files.
    """
    import pandas as pd
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
    print(f"Loading manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    # Check for split column if present
    if "unified_split" in df.columns:
        df = df[df["unified_split"] == "test"].copy()
    elif "split" in df.columns:
        df = df[df["split"] == "test"].copy()
        
    if limit is not None and limit > 0:
        df = df.head(limit).copy()
        
    print(f"Processing {len(df)} manifest entries...")
    
    # Determine column names
    id_col = None
    for cand in ["clip_id", "video_id", "id", "sample_id"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = df.columns[0]
        
    video_col = None
    for cand in ["video_path", "path", "file_path", "video_file"]:
        if cand in df.columns:
            video_col = cand
            break
            
    npz_col = None
    for cand in ["npz_path", "feature_path", "landmarks_path"]:
        if cand in df.columns:
            npz_col = cand
            break
            
    label_col = None
    for cand in ["normalized_label", "gloss", "label", "target"]:
        if cand in df.columns:
            label_col = cand
            break
            
    from scripts.prepare_sequences import resample_sequence
    from scripts.live_inference import LiveTemporalPreprocessor, LiveLandmarkExtractor
    import cv2
    
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    extractor = None  # Lazy initialize if needed
    
    predictions = []
    correct_top1 = 0
    correct_top5 = 0
    evaluated_with_gt = 0
    
    for idx, row in df.iterrows():
        clip_id = str(row[id_col])
        npz_file = str(row[npz_col]) if npz_col and pd.notna(row[npz_col]) else None
        video_file = str(row[video_col]) if video_col and pd.notna(row[video_col]) else None
        
        # 1. Try loading from NPZ if available
        features_263 = None
        if npz_file and os.path.exists(npz_file):
            npz_data = np.load(npz_file)
            if "flat_features" in npz_data:
                feat_134 = npz_data["flat_features"]
            elif "X" in npz_data:
                feat_134 = npz_data["X"]
            else:
                feat_134 = npz_data[list(npz_data.keys())[0]]
            resampled_134 = resample_sequence(feat_134, target_len=64)
            features_263 = preprocessor.process_window(resampled_134)
            
        # 2. Extract from video if NPZ not available
        elif video_file and os.path.exists(video_file):
            if extractor is None:
                extractor = LiveLandmarkExtractor()
            cap = cv2.VideoCapture(video_file)
            extractor.reset_tracking()
            frames_134 = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                flat_134, _, _ = extractor.process_frame(frame)
                frames_134.append(flat_134)
            cap.release()
            
            if len(frames_134) == 0:
                frames_134 = [np.zeros(134, dtype=np.float32) for _ in range(64)]
            seq_arr = np.array(frames_134, dtype=np.float32)
            resampled_134 = resample_sequence(seq_arr, target_len=64)
            features_263 = preprocessor.process_window(resampled_134)
        else:
            raise FileNotFoundError(f"Neither NPZ ({npz_file}) nor video ({video_file}) found for {clip_id}.")
            
        res = runtime.predict(features_263)
        pred_gloss = str(res["predicted_class"]).upper()
        conf = round(float(res["confidence"]), 4)
        top5_glosses = [str(item["class_name"]).upper() for item in res["top_5"]]
        lat_ms = round(float(res["latency_ms"]), 2)
        
        predictions.append({
            "clip_id": clip_id,
            "prediction": pred_gloss,
            "confidence": conf,
            "top5": top5_glosses,
            "latency_ms": lat_ms
        })
        
        if label_col and pd.notna(row[label_col]):
            gt = str(row[label_col]).strip().upper()
            evaluated_with_gt += 1
            if pred_gloss == gt:
                correct_top1 += 1
            if gt in top5_glosses:
                correct_top5 += 1
                
    if extractor is not None:
        extractor.close()
        
    metrics = None
    if evaluated_with_gt > 0:
        metrics = {
            "top1_accuracy": correct_top1 / evaluated_with_gt,
            "top5_accuracy": correct_top5 / evaluated_with_gt,
            "top1_count": correct_top1,
            "top5_count": correct_top5,
            "total_samples": evaluated_with_gt
        }
    return predictions, metrics


def process_video_directory(
    video_dir: str,
    scaler_path: str,
    runtime: EchoGestureRuntime,
    limit: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], None]:
    """
    Generate predictions from raw video files in a directory.
    """
    import cv2
    from scripts.prepare_sequences import resample_sequence
    from scripts.live_inference import LiveTemporalPreprocessor, LiveLandmarkExtractor
    
    p_dir = Path(video_dir)
    if not p_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted([f for f in p_dir.iterdir() if f.suffix.lower() in video_exts])
    
    if limit is not None and limit > 0:
        video_files = video_files[:limit]
        
    print(f"Found {len(video_files)} video files in {video_dir}...")
    
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    
    predictions = []
    for f in video_files:
        clip_id = f.stem
        cap = cv2.VideoCapture(str(f))
        extractor.reset_tracking()
        frames_134 = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            flat_134, _, _ = extractor.process_frame(frame)
            frames_134.append(flat_134)
        cap.release()
        
        if len(frames_134) == 0:
            frames_134 = [np.zeros(134, dtype=np.float32) for _ in range(64)]
        seq_arr = np.array(frames_134, dtype=np.float32)
        resampled_134 = resample_sequence(seq_arr, target_len=64)
        features_263 = preprocessor.process_window(resampled_134)
        
        res = runtime.predict(features_263)
        pred_gloss = str(res["predicted_class"]).upper()
        conf = round(float(res["confidence"]), 4)
        top5_glosses = [str(item["class_name"]).upper() for item in res["top_5"]]
        lat_ms = round(float(res["latency_ms"]), 2)
        
        predictions.append({
            "clip_id": clip_id,
            "prediction": pred_gloss,
            "confidence": conf,
            "top5": top5_glosses,
            "latency_ms": lat_ms
        })
        
    extractor.close()
    return predictions, None


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EchoGesture: Official Predictions Generator (predictions.jsonl)")
    print("Track 02: On-Device Real-Time ASL Recognition System")
    print("=" * 70)
    
    # 1. Initialize Runtime with Frozen INT8 Quantized Model
    print(f"Loading TFLite Model:    {args.model}")
    print(f"Loading Label Encoder:   {args.label_encoder}")
    runtime = EchoGestureRuntime(
        model_path=args.model,
        label_encoder_path=args.label_encoder,
        num_threads=args.num_threads
    )
    print(f"Loaded {runtime.num_classes} vocabulary classes.")
    
    # 2. Select Ingestion Pipeline
    t_start = time.perf_counter()
    if args.manifest and os.path.exists(args.manifest):
        print(f"\nExecution Mode: Manifest CSV Evaluation ({args.manifest})")
        predictions, metrics = process_manifest_csv(
            manifest_path=args.manifest,
            scaler_path=args.scaler,
            runtime=runtime,
            limit=args.limit
        )
    elif args.video_dir and os.path.exists(args.video_dir):
        print(f"\nExecution Mode: Video Directory Evaluation ({args.video_dir})")
        predictions, metrics = process_video_directory(
            video_dir=args.video_dir,
            scaler_path=args.scaler,
            runtime=runtime,
            limit=args.limit
        )
    elif args.sequences and os.path.exists(args.sequences):
        print(f"\nExecution Mode: Pre-Extracted Sequences (.npz) ({args.sequences})")
        predictions, metrics = process_sequences_npz(
            sequences_path=args.sequences,
            runtime=runtime,
            limit=args.limit
        )
    else:
        raise ValueError(
            "No valid input source provided. Please specify --sequences <file.npz>, "
            "--manifest <file.csv>, or --video-dir <directory>."
        )
        
    elapsed = time.perf_counter() - t_start
    num_predictions = len(predictions)
    print(f"\nInference completed in {elapsed:.2f}s across {num_predictions} clips.")
    
    if metrics:
        print("\n--- Empirical Benchmark Validation ---")
        print(f"Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}% ({metrics['top1_count']}/{metrics['total_samples']})")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}% ({metrics['top5_count']}/{metrics['total_samples']})")
        
    # 3. Output to JSON Lines file
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting predictions to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for record in predictions:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")
            
    print(f"Successfully generated {num_predictions} lines in {out_file}.")
    
    # 4. Verification of Output Integrity
    print("\n--- Verifying Output Integrity ---")
    seen_ids = set()
    valid_classes = {c.upper() for c in runtime.idx2label.values()}
    line_count = 0
    with open(out_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_count += 1
            entry = json.loads(line)
            # Schema verification
            for key in ["clip_id", "prediction", "confidence", "top5", "latency_ms"]:
                assert key in entry, f"Missing key '{key}' on line {idx}"
            assert isinstance(entry["clip_id"], str) and len(entry["clip_id"]) > 0
            assert entry["clip_id"] not in seen_ids, f"Duplicate clip_id: {entry['clip_id']}"
            seen_ids.add(entry["clip_id"])
            assert entry["prediction"] in valid_classes, f"Invalid class: {entry['prediction']}"
            assert isinstance(entry["top5"], list) and len(entry["top5"]) == 5
            for c in entry["top5"]:
                assert c in valid_classes, f"Invalid top-5 class: {c}"
            assert 0.0 <= entry["confidence"] <= 1.0, f"Confidence out of range: {entry['confidence']}"
            assert entry["latency_ms"] >= 0.0, f"Latency invalid: {entry['latency_ms']}"
            
    print(f"Integrity PASS: Exactly {line_count} valid JSON Lines with unique clip IDs and valid vocabulary.")


if __name__ == "__main__":
    main()
