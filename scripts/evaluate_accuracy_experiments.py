"""
EchoGesture Phase 5A: Final Test Evaluation for Winning Improved Model
Evaluates the winning model selected on validation data strictly ONCE on the held-out Test partition.

Generates:
- Formal comparison vs. Original Baseline (52.79% Top-1, 82.23% Top-5, 0.4700 Macro F1)
- High-resolution normalized confusion matrix (reports/figures/confusion_matrix_v5a.png)
- Per-class metrics table (reports/per_class_v5a_metrics.csv)
- Structured evaluation JSON (reports/phase5a_final_test_evaluation.json)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import tensorflow as tf
import keras

from scripts.augment_sequences import batch_compute_velocity


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Phase 5A Improved Model: Test Set Confusion Matrix"
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(all="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"}
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Sign Class", fontsize=11)
    plt.ylabel("True Sign Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")


def evaluate_winning_model(
    selection_json: str = "reports/phase5a_winning_selection.json",
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== EchoGesture Phase 5A: Final Held-Out Test Evaluation ===")
    
    # 1. Load Winning Model Selection
    with open(selection_json, "r", encoding="utf-8") as f:
        winner = json.load(f)
        
    exp_id = winner["exp_id"]
    exp_name = winner["exp_name"]
    use_velocity = winner["use_velocity"]
    model_path = "models/checkpoints/best_improved_model_v5a.keras"
    
    print(f"Winning Configuration: {exp_id} ({exp_name})")
    print(f"Using Velocity Features: {use_velocity}")
    print(f"Model Checkpoint: {model_path}")
    
    # 2. Load Model
    model = keras.models.load_model(model_path)
    model.summary()
    param_count = model.count_params()
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # 3. Load Test Data & Label Encoder
    p_data = Path(data_dir)
    with open(p_data / "label_encoder.json", "r", encoding="utf-8") as f:
        enc = json.load(f)
    class_names = enc["classes"]
    num_classes = enc["num_classes"]
    
    test_npz = np.load(p_data / "test_sequences.npz")
    X_test_134 = test_npz["X"]
    y_test = test_npz["y"]
    test_vids = test_npz["video_ids"]
    
    # Apply velocity transformation if required by winning model
    if use_velocity:
        print("Computing velocity features for test sequences...")
        X_test = batch_compute_velocity(X_test_134)
    else:
        X_test = X_test_134
        
    print(f"Test tensor shape: {X_test.shape}, labels: {y_test.shape}")
    
    # 4. Single-Pass Inference on Test Set
    t0 = time.perf_counter()
    test_probs = model.predict(X_test, batch_size=32, verbose=0)
    eval_time = time.perf_counter() - t0
    y_pred = np.argmax(test_probs, axis=1)
    
    # 5. Compute Metrics
    all_labels = list(range(num_classes))
    top1 = float(accuracy_score(y_test, y_pred))
    
    top5_preds = np.argsort(test_probs, axis=1)[:, -5:]
    top5 = float(np.mean([y_test[i] in top5_preds[i] for i in range(len(y_test))]))
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=all_labels, average="weighted", zero_division=0
    )
    
    # Per-Class Metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_test, y_pred, labels=all_labels, average=None, zero_division=0
    )
    
    per_class_records = []
    for i, name in enumerate(class_names):
        per_class_records.append({
            "class_id": i,
            "class_name": name,
            "support": int(support_class[i]),
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i])
        })
    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv = out_path / "per_class_v5a_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"Saved per-class test metrics to {per_class_csv}")
    
    # Confusion Matrix
    cm_path = fig_dir / "confusion_matrix_v5a.png"
    plot_confusion_matrix(y_test, y_pred, class_names, str(cm_path), title=f"Phase 5A {exp_id}: Test Confusion Matrix")
    
    # CPU Latency Benchmarking (100 runs)
    print("Benchmarking CPU inference latency...")
    input_sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)
    for _ in range(15):
        _ = model(input_sample, training=False)
    timings = []
    for _ in range(100):
        t_start = time.perf_counter()
        _ = model(input_sample, training=False)
        timings.append((time.perf_counter() - t_start) * 1000.0)
    latency_stats = {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings))
    }
    
    # Baseline Comparison
    baseline_top1 = 0.5279
    baseline_top5 = 0.8223
    baseline_macro_f1 = 0.4700
    
    delta_top1 = top1 - baseline_top1
    delta_top5 = top5 - baseline_top5
    delta_macro_f1 = f1_macro - baseline_macro_f1
    
    results = {
        "winning_configuration": exp_id,
        "experiment_name": exp_name,
        "feature_dim": int(X_test.shape[-1]),
        "parameter_count": param_count,
        "model_size_mb": model_size_mb,
        "latency_stats": latency_stats,
        "test_results": {
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_wt),
            "test_sample_count": len(y_test)
        },
        "baseline_comparison": {
            "baseline_top1": baseline_top1,
            "baseline_top5": baseline_top5,
            "baseline_macro_f1": baseline_macro_f1,
            "delta_top1": delta_top1,
            "delta_top5": delta_top5,
            "delta_macro_f1": delta_macro_f1,
            "is_improved": bool(delta_macro_f1 > 0 or delta_top1 > 0)
        }
    }
    
    json_path = out_path / "phase5a_final_test_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation JSON to {json_path}")
    
    print("\n=========================================================================================")
    print("                     PHASE 5A FINAL HELD-OUT TEST RESULTS COMPARISON                     ")
    print("=========================================================================================")
    print(f" Metric             | Baseline (Phase 4) | Phase 5A ({exp_id}) | Delta")
    print(f" -------------------+--------------------+--------------------+-------------------------")
    print(f" Top-1 Accuracy     | {baseline_top1*100:6.2f}%            | {top1*100:6.2f}%            | {delta_top1*100:+6.2f}%")
    print(f" Top-5 Accuracy     | {baseline_top5*100:6.2f}%            | {top5*100:6.2f}%            | {delta_top5*100:+6.2f}%")
    print(f" Macro F1-Score     | {baseline_macro_f1:6.4f}             | {f1_macro:6.4f}             | {delta_macro_f1:+6.4f}")
    print(f" CPU Latency (Mean) | 17.57 ms           | {latency_stats['mean_ms']:5.2f} ms           | {latency_stats['mean_ms'] - 17.57:+5.2f} ms")
    print(f" Parameter Count    | 87,336             | {param_count:,}            | {param_count - 87336:+,}")
    print("=========================================================================================")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 5A Test Evaluation")
    parser.add_argument("--selection", type=str, default="reports/phase5a_winning_selection.json")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    
    evaluate_winning_model(args.selection, args.data_dir, args.output_dir)
