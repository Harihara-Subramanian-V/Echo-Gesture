"""
EchoGesture Phase 4: Baseline Model Evaluation Pipeline
Evaluates best trained checkpoint on Validation and held-out Test partitions.

Computes:
- Top-1 Accuracy & Top-5 Accuracy
- Macro & Weighted F1, Precision, Recall (zero_division=0 for 39-class val split)
- Per-class classification report
- High-resolution normalized confusion matrix plot
- CPU inference latency benchmarking (ms/sample) and model footprint (MB)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
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
    confusion_matrix,
    classification_report
)

import tensorflow as tf
import keras


def load_evaluation_data(data_dir: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """Load preprocessed validation and test datasets and label encoder."""
    p = Path(data_dir)
    
    with open(p / "label_encoder.json", "r", encoding="utf-8") as f:
        encoder_data = json.load(f)
        
    val_npz = np.load(p / "val_sequences.npz")
    test_npz = np.load(p / "test_sequences.npz")
    
    splits = {
        "val": (val_npz["X"], val_npz["y"], val_npz["video_ids"]),
        "test": (test_npz["X"], test_npz["y"], test_npz["video_ids"])
    }
    
    return splits, encoder_data


def benchmark_inference_latency(model: keras.Model, sample: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
    """
    Measure single-sample inference latency on CPU over repeated trials.
    """
    # Ensure shape is (1, 64, 134)
    input_tensor = np.expand_dims(sample, axis=0).astype(np.float32)
    
    # Warmup runs
    for _ in range(15):
        _ = model(input_tensor, training=False)
        
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(input_tensor, training=False)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # ms
        
    timings = np.array(timings)
    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings))
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute Top-1, Top-5, Macro/Weighted F1, and per-class metrics.
    Handles classes with 0 support (e.g. 'thanksgiving' in val) via zero_division=0.
    """
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = len(class_names)
    all_labels = list(range(num_classes))
    
    # Top-1 accuracy
    top1 = float(accuracy_score(y_true, y_pred))
    
    # Top-5 accuracy
    try:
        top5 = float(top_k_accuracy_score(y_true, y_prob, k=5, labels=all_labels))
    except Exception:
        # Fallback manual top-5 calculation
        top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
        matches = [y_true[i] in top5_preds[i] for i in range(len(y_true))]
        top5 = float(np.mean(matches))
        
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )
    
    per_class_list = []
    for i, name in enumerate(class_names):
        per_class_list.append({
            "class_id": i,
            "class_name": name,
            "support": int(support_class[i]),
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i])
        })
        
    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_list
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Test Set Confusion Matrix (Normalized)"
):
    """Plot and save high-resolution normalized confusion matrix heatmap."""
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
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")


def evaluate_baseline(
    model_path: str = "models/checkpoints/best_baseline_model.keras",
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports"
):
    """Execute complete evaluation suite."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== EchoGesture Baseline Model Evaluation ===")
    print(f"Loading checkpoint: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}!")
        
    model = keras.models.load_model(model_path)
    model.summary()
    param_count = model.count_params()
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model Parameters: {param_count:,}")
    print(f"Model File Size:  {model_size_mb:.2f} MB")
    
    # Load dataset
    splits, encoder_data = load_evaluation_data(data_dir)
    class_names = encoder_data["classes"]
    num_classes = encoder_data["num_classes"]
    
    results = {
        "model_path": model_path,
        "parameter_count": param_count,
        "model_size_mb": float(model_size_mb),
        "num_classes": num_classes,
        "splits": {}
    }
    
    # Evaluate Validation Split
    print("\n--- Evaluating Validation Split (262 clips, 39 active classes) ---")
    X_val, y_val, val_ids = splits["val"]
    val_probs = model.predict(X_val, batch_size=32, verbose=0)
    val_metrics = compute_metrics(y_val, val_probs, class_names)
    results["splits"]["validation"] = {
        "sample_count": len(y_val),
        "active_classes": int(len(np.unique(y_val))),
        "metrics": val_metrics
    }
    print(f"  Val Top-1 Accuracy: {val_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Val Top-5 Accuracy: {val_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Val Macro F1:       {val_metrics['macro_f1']:.4f}")
    print(f"  Val Weighted F1:    {val_metrics['weighted_f1']:.4f}")
    
    # Evaluate Test Split
    print("\n--- Evaluating Held-Out Test Split (197 clips, 40 classes) ---")
    X_test, y_test, test_ids = splits["test"]
    test_probs = model.predict(X_test, batch_size=32, verbose=0)
    test_metrics = compute_metrics(y_test, test_probs, class_names)
    results["splits"]["test"] = {
        "sample_count": len(y_test),
        "active_classes": int(len(np.unique(y_test))),
        "metrics": test_metrics
    }
    print(f"  Test Top-1 Accuracy: {test_metrics['top1_accuracy']*100:.2f}%")
    print(f"  Test Top-5 Accuracy: {test_metrics['top5_accuracy']*100:.2f}%")
    print(f"  Test Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1:    {test_metrics['weighted_f1']:.4f}")
    
    # Save per-class metrics to CSV
    per_class_df = pd.DataFrame(test_metrics["per_class"])
    per_class_csv = out_path / "per_class_baseline_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"\nSaved per-class test metrics to {per_class_csv}")
    
    # Plot Confusion Matrix for Test
    y_test_pred = np.argmax(test_probs, axis=1)
    cm_path = fig_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_test_pred, class_names, str(cm_path))
    
    # Benchmark CPU Inference Latency
    print("\n--- Benchmarking CPU Inference Latency ---")
    latency_stats = benchmark_inference_latency(model, X_test[0], num_runs=100)
    results["inference_latency"] = latency_stats
    print(f"  Mean Latency:   {latency_stats['mean_ms']:.2f} ms")
    print(f"  Median Latency: {latency_stats['median_ms']:.2f} ms")
    print(f"  95th Pct:       {latency_stats['p95_ms']:.2f} ms")
    print(f"  Min / Max:      {latency_stats['min_ms']:.2f} / {latency_stats['max_ms']:.2f} ms")
    
    # Save full JSON report
    metrics_json = out_path / "baseline_evaluation_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured metrics to {metrics_json}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Baseline Evaluation")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_baseline_model.keras")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    
    evaluate_baseline(args.model, args.data_dir, args.output_dir)
