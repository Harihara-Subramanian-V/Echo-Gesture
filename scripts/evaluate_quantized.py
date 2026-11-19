"""
EchoGesture Phase 5B: Quantized Model Evaluation Pipeline
Runs comparative evaluation between original Float32 EXP_D model and Quantized INT8 TFLite model.

Evaluates on:
- Unchanged Validation partition (262 clips, 39 active classes)
- Untouched held-out Test partition (197 clips, 40 classes)

Reports:
- Top-1 & Top-5 accuracy
- Macro & Weighted F1
- Prediction agreement (% matching predictions)
- Quantization accuracy degradation / retention in percentage points
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support
)

import tensorflow as tf
import keras

from scripts.augment_sequences import batch_compute_velocity


def run_int8_tflite_inference(
    interpreter: tf.lite.Interpreter,
    X_float: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute full-integer INT8 inference sample-by-sample using TFLite Interpreter.
    
    Transforms:
        Input: Quantizes float32 features to int8 via S_in and Z_in
        Output: Dequantizes int8 logits to float32 probabilities via S_out and Z_out
    """
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    in_idx = in_details["index"]
    out_idx = out_details["index"]
    
    s_in, z_in = in_details["quantization"]
    s_out, z_out = out_details["quantization"]
    
    N = len(X_float)
    num_classes = out_details["shape"][-1]
    
    int8_probs = np.zeros((N, num_classes), dtype=np.float32)
    int8_preds = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        sample = X_float[i]
        
        # Affine Quantization: q = clip(round(x / S) + Z, -128, 127)
        q_in = np.clip(np.round(sample / s_in) + z_in, -128, 127).astype(np.int8)
        q_in_batch = np.expand_dims(q_in, axis=0)
        
        interpreter.set_tensor(in_idx, q_in_batch)
        interpreter.invoke()
        
        q_out = interpreter.get_tensor(out_idx)[0]  # (40,) int8
        
        # Output dequantization: x = (q - Z) * S
        prob_float = (q_out.astype(np.float32) - float(z_out)) * float(s_out)
        
        int8_probs[i] = prob_float
        int8_preds[i] = int(np.argmax(q_out))
        
    return int8_probs, int8_preds


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 40
) -> Dict[str, float]:
    """Compute Top-1, Top-5, Macro/Weighted F1 with zero_division=0."""
    y_pred = np.argmax(y_prob, axis=1)
    all_labels = list(range(num_classes))
    
    top1 = float(accuracy_score(y_true, y_pred))
    
    # Top-5 Accuracy
    top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
    top5 = float(np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))]))
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="weighted", zero_division=0
    )
    
    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt)
    }


def compare_float_vs_int8(
    float_model_path: str = "models/checkpoints/best_improved_model_v5a.keras",
    int8_tflite_path: str = "models/quantized/echogesture_exp_d_int8.tflite",
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports"
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=== EchoGesture Phase 5B: Float32 vs. INT8 Accuracy Evaluation ===")
    
    # 1. Load Float Model & INT8 Interpreter
    print(f"Loading Float32 model: {float_model_path}")
    float_model = keras.models.load_model(float_model_path)
    
    print(f"Loading INT8 TFLite model: {int8_tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=int8_tflite_path)
    interpreter.allocate_tensors()
    
    # 2. Load Datasets
    p_data = Path(data_dir)
    with open(p_data / "label_encoder.json", "r", encoding="utf-8") as f:
        enc = json.load(f)
    num_classes = enc["num_classes"]
    class_names = enc["classes"]
    
    val_npz = np.load(p_data / "val_sequences.npz")
    test_npz = np.load(p_data / "test_sequences.npz")
    
    # Exact 263-D velocity features for validation and test
    print("Computing 263-D velocity features for Validation and Test...")
    X_val_263 = batch_compute_velocity(val_npz["X"])
    y_val = val_npz["y"]
    
    X_test_263 = batch_compute_velocity(test_npz["X"])
    y_test = test_npz["y"]
    
    partitions = [
        ("Validation", X_val_263, y_val),
        ("Held-Out Test", X_test_263, y_test)
    ]
    
    summary_rows = []
    full_results = {
        "float_model_path": float_model_path,
        "int8_tflite_path": int8_tflite_path,
        "partitions": {}
    }
    
    for part_name, X_part, y_part in partitions:
        print(f"\n--- Evaluating {part_name} Partition ({len(y_part)} samples) ---")
        
        # A. Float32 Model Inference
        float_probs = float_model.predict(X_part, batch_size=32, verbose=0)
        float_preds = np.argmax(float_probs, axis=1)
        float_metrics = compute_classification_metrics(y_part, float_probs, num_classes)
        
        # B. INT8 Model Inference
        int8_probs, int8_preds = run_int8_tflite_inference(interpreter, X_part)
        int8_metrics = compute_classification_metrics(y_part, int8_probs, num_classes)
        
        # C. Prediction Agreement
        agreement = float(np.mean(float_preds == int8_preds))
        
        # Deltas
        delta_top1 = int8_metrics["top1_accuracy"] - float_metrics["top1_accuracy"]
        delta_top5 = int8_metrics["top5_accuracy"] - float_metrics["top5_accuracy"]
        delta_macro_f1 = int8_metrics["macro_f1"] - float_metrics["macro_f1"]
        
        print(f"  Float32 Top-1: {float_metrics['top1_accuracy']*100:.2f}% | INT8 Top-1: {int8_metrics['top1_accuracy']*100:.2f}% (Delta: {delta_top1*100:+.2f}%)")
        print(f"  Float32 Top-5: {float_metrics['top5_accuracy']*100:.2f}% | INT8 Top-5: {int8_metrics['top5_accuracy']*100:.2f}% (Delta: {delta_top5*100:+.2f}%)")
        print(f"  Float32 Macro F1: {float_metrics['macro_f1']:.4f} | INT8 Macro F1: {int8_metrics['macro_f1']:.4f} (Delta: {delta_macro_f1:+.4f})")
        print(f"  Prediction Agreement: {agreement*100:.2f}%")
        
        summary_rows.append({
            "partition": part_name,
            "samples": len(y_part),
            "float_top1": float_metrics["top1_accuracy"],
            "int8_top1": int8_metrics["top1_accuracy"],
            "delta_top1_pct": delta_top1 * 100.0,
            "float_top5": float_metrics["top5_accuracy"],
            "int8_top5": int8_metrics["top5_accuracy"],
            "delta_top5_pct": delta_top5 * 100.0,
            "float_macro_f1": float_metrics["macro_f1"],
            "int8_macro_f1": int8_metrics["macro_f1"],
            "delta_macro_f1": delta_macro_f1,
            "prediction_agreement": agreement
        })
        
        full_results["partitions"][part_name.lower().replace(" ", "_")] = {
            "sample_count": len(y_part),
            "float_metrics": float_metrics,
            "int8_metrics": int8_metrics,
            "deltas": {
                "top1_delta": delta_top1,
                "top5_delta": delta_top5,
                "macro_f1_delta": delta_macro_f1
            },
            "prediction_agreement": agreement
        }
        
    df_summary = pd.DataFrame(summary_rows)
    csv_out = out_path / "float_vs_int8_comparison.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved comparison summary table to {csv_out}")
    
    json_out = out_path / "int8_quantization_evaluation.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved structured evaluation JSON to {json_out}")
    
    print("\n=========================================================================================")
    print("                    FLOAT32 VS. FULL INT8 QUANTIZED ACCURACY COMPARISON                   ")
    print("=========================================================================================")
    print(df_summary[[
        "partition", "float_top1", "int8_top1", "delta_top1_pct", "float_top5", "int8_top5", "float_macro_f1", "int8_macro_f1", "prediction_agreement"
    ]].to_string(index=False))
    print("=========================================================================================")
    
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Float vs INT8 Evaluation")
    parser.add_argument("--float_model", type=str, default="models/checkpoints/best_improved_model_v5a.keras")
    parser.add_argument("--int8_tflite", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    
    compare_float_vs_int8(args.float_model, args.int8_tflite, args.data_dir, args.output_dir)
