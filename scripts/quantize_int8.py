"""
EchoGesture Phase 5B: Strict Full INT8 Post-Training Quantization
Converts trained Keras model to production TFLite format with full integer quantization.

Strict Protocols:
- Representative calibration dataset derived strictly from training partition (Zero Data Leakage)
- Full INT8 input and output tensors
- Pure TFLite builtins only (no SELECT_TF_OPS fallback)
- Structural verification of integer inference graph
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from scripts.augment_sequences import compute_velocity_features


def build_representative_dataset(
    train_npz_path: str,
    manifest_csv_path: str,
    label_encoder_path: str,
    samples_per_class: int = 4,
    random_seed: int = 42
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Select stratified calibration samples strictly from the training partition.
    
    Ensures:
    - Balanced representation across all 40 sign vocabulary classes
    - Diverse training signers and sequence durations
    - Exact 263-dimensional feature representation (134 base + 129 velocity)
    """
    np.random.seed(random_seed)
    
    # Load training sequences and metadata
    train_npz = np.load(train_npz_path)
    X_train_134 = train_npz["X"]
    y_train = train_npz["y"]
    vids_train = train_npz["video_ids"]
    
    with open(label_encoder_path, "r", encoding="utf-8") as f:
        enc = json.load(f)
    idx2label = {int(k): v for k, v in enc["idx2label"].items()}
    num_classes = enc["num_classes"]
    
    # Load feature manifest for detailed signer and duration metadata
    fm = pd.read_csv(manifest_csv_path).set_index("video_id")
    
    calib_indices = []
    for c_id in range(num_classes):
        matching_indices = np.where(y_train == c_id)[0]
        n_select = min(samples_per_class, len(matching_indices))
        selected = np.random.choice(matching_indices, size=n_select, replace=False)
        calib_indices.extend(selected)
        
    calib_indices = sorted(calib_indices)
    
    calib_records = []
    calibration_tensors_263 = []
    
    for i, idx in enumerate(calib_indices):
        vid = str(vids_train[idx])
        c_id = int(y_train[idx])
        c_name = idx2label[c_id]
        
        # Exact EXP_D 263-D feature computation
        seq_134 = X_train_134[idx]
        seq_263 = compute_velocity_features(seq_134).astype(np.float32)
        calibration_tensors_263.append(seq_263)
        
        row = fm.loc[vid] if vid in fm.index else {}
        calib_records.append({
            "calibration_id": i,
            "train_array_index": int(idx),
            "video_id": vid,
            "signer_id": row.get("signer_id", "unknown"),
            "class_name": c_name,
            "class_id": c_id,
            "original_frames": int(row.get("total_frames", 64)),
            "fps": float(row.get("fps", 30.0)),
            "any_hand_detection_rate": float(row.get("any_hand_detection_rate", 1.0)),
            "both_hands_detection_rate": float(row.get("both_hands_detection_rate", 0.0))
        })
        
    df_manifest = pd.DataFrame(calib_records)
    return calibration_tensors_263, df_manifest


def convert_model_to_int8(
    model_path: str,
    calibration_tensors: List[np.ndarray],
    output_tflite_path: str
) -> bytes:
    """
    Perform full INT8 quantization on SavedModel artifact.
    """
    print(f"Loading Keras checkpoint: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Representative dataset generator
    def representative_dataset_gen() -> Generator[List[np.ndarray], None, None]:
        for tensor in calibration_tensors:
            # Batch dimension of 1: shape (1, 64, 263)
            yield [np.expand_dims(tensor, axis=0).astype(np.float32)]
            
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")
        print(f"Exporting intermediate SavedModel to {saved_model_dir}...")
        model.export(saved_model_dir)
        
        print("Configuring full-integer TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        
        # Enforce full INT8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        print("Executing post-training INT8 quantization...")
        quantized_tflite_bytes = converter.convert()
        
    Path(output_tflite_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(quantized_tflite_bytes)
        
    print(f"Successfully saved INT8 quantized model to {output_tflite_path}")
    return quantized_tflite_bytes


def verify_quantized_model_structure(tflite_path: str) -> Dict[str, Any]:
    """
    Perform comprehensive structural inspection of the TFLite binary.
    """
    print("\n--- Structural Verification of Quantized TFLite Model ---")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    in_tensor = in_details[0]
    out_tensor = out_details[0]
    
    in_shape = list(in_tensor["shape"])
    in_dtype = str(in_tensor["dtype"].__name__)
    in_scale, in_zero_point = in_tensor["quantization"]
    
    out_shape = list(out_tensor["shape"])
    out_dtype = str(out_tensor["dtype"].__name__)
    out_scale, out_zero_point = out_tensor["quantization"]
    
    # Count data types across all tensors
    dtype_counts = {}
    for t in tensor_details:
        dt = str(t["dtype"].__name__)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        
    file_size_bytes = os.path.getsize(tflite_path)
    file_size_kb = file_size_bytes / 1024.0
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)
    
    print(f"  Input Shape:         {in_shape}")
    print(f"  Input Data Type:     {in_dtype} (scale={in_scale:.6f}, zero_point={in_zero_point})")
    print(f"  Output Shape:        {out_shape}")
    print(f"  Output Data Type:    {out_dtype} (scale={out_scale:.6f}, zero_point={out_zero_point})")
    print(f"  Tensor Dtype Counts: {dtype_counts}")
    print(f"  File Size:           {file_size_kb:.2f} KB ({file_size_mb:.4f} MB)")
    
    # Assertions
    assert in_shape == [1, 64, 263], f"Expected input shape [1, 64, 263], got {in_shape}"
    assert in_dtype == "int8", f"Input tensor is not int8! Got {in_dtype}"
    assert out_shape == [1, 40], f"Expected output shape [1, 40], got {out_shape}"
    assert out_dtype == "int8", f"Output tensor is not int8! Got {out_dtype}"
    assert file_size_mb < 20.0, f"Model size ({file_size_mb:.2f} MB) exceeds 20 MB budget!"
    
    print("Structural verification: PASSED (Strict Full INT8 Integer Graph)")
    
    return {
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_kb,
        "file_size_mb": file_size_mb,
        "input_tensor": {
            "name": in_tensor["name"],
            "shape": in_shape,
            "dtype": in_dtype,
            "scale": float(in_scale),
            "zero_point": int(in_zero_point)
        },
        "output_tensor": {
            "name": out_tensor["name"],
            "shape": out_shape,
            "dtype": out_dtype,
            "scale": float(out_scale),
            "zero_point": int(out_zero_point)
        },
        "dtype_counts": dtype_counts,
        "total_tensors": len(tensor_details)
    }


def run_quantization_pipeline(
    model_path: str = "models/checkpoints/best_improved_model_v5a.keras",
    train_npz: str = "data/processed/sequences_v3/train_sequences.npz",
    manifest_csv: str = "data/processed/landmarks_v3/feature_manifest.csv",
    label_encoder: str = "data/processed/sequences_v3/label_encoder.json",
    output_tflite: str = "models/quantized/echogesture_exp_d_int8.tflite",
    manifest_out: str = "reports/int8_calibration_manifest.csv"
) -> Dict[str, Any]:
    print("=== EchoGesture Phase 5B: Strict INT8 Quantization ===")
    
    # 1. Build Representative Calibration Dataset
    print("\nStep 1: Building Representative Calibration Dataset...")
    calib_tensors, df_calib = build_representative_dataset(
        train_npz_path=train_npz,
        manifest_csv_path=manifest_csv,
        label_encoder_path=label_encoder,
        samples_per_class=4,
        random_seed=42
    )
    print(f"  Selected {len(calib_tensors)} calibration samples across {df_calib['class_name'].nunique()} classes.")
    print(f"  Signers represented: {df_calib['signer_id'].nunique()} unique training signers.")
    
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    df_calib.to_csv(manifest_out, index=False)
    print(f"  Saved calibration manifest to {manifest_out}")
    
    # 2. Perform Full-Integer Quantization
    print("\nStep 2: Executing Full-Integer Quantization...")
    _ = convert_model_to_int8(
        model_path=model_path,
        calibration_tensors=calib_tensors,
        output_tflite_path=output_tflite
    )
    
    # 3. Structural Inspection
    print("\nStep 3: Verifying Integer Model Structure...")
    structure_info = verify_quantized_model_structure(output_tflite)
    
    return {
        "status": "SUCCESS",
        "calibration_samples_count": len(calib_tensors),
        "calibration_classes_count": int(df_calib["class_name"].nunique()),
        "output_tflite_path": output_tflite,
        "structure": structure_info
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture INT8 Quantization")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_improved_model_v5a.keras")
    parser.add_argument("--train_npz", type=str, default="data/processed/sequences_v3/train_sequences.npz")
    parser.add_argument("--manifest_csv", type=str, default="data/processed/landmarks_v3/feature_manifest.csv")
    parser.add_argument("--label_encoder", type=str, default="data/processed/sequences_v3/label_encoder.json")
    parser.add_argument("--output_tflite", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--manifest_out", type=str, default="reports/int8_calibration_manifest.csv")
    args = parser.parse_args()
    
    run_quantization_pipeline(
        model_path=args.model,
        train_npz=args.train_npz,
        manifest_csv=args.manifest_csv,
        label_encoder=args.label_encoder,
        output_tflite=args.output_tflite,
        manifest_out=args.manifest_out
    )
