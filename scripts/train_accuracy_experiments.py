"""
EchoGesture Phase 5A: Controlled Accuracy Improvement Experiments
Runs systematic ablation study comparing 4 configurations under identical conditions:

- Exp A: Original Compact TCN Baseline (134-D, no augmentation)
- Exp B: Compact TCN + Velocity Features (263-D, no augmentation)
- Exp C: Compact TCN + Training-Only Augmentation (134-D, dynamic augmentation)
- Exp D: Compact TCN + Velocity Features + Augmentation (263-D, dynamic augmentation)

Strict Protocol:
- Identical random seed (42) and identical signer-independent splits
- Model selection is governed STRICTLY by validation split performance
- Zero data leakage: augmentation applied exclusively during training
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_recall_fscore_support

import tensorflow as tf
import keras
from keras import layers, callbacks, optimizers, losses, metrics

from scripts.augment_sequences import (
    DynamicAugmentedSequence,
    compute_velocity_features,
    batch_compute_velocity
)


def set_random_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_compact_tcn(
    sequence_length: int = 64,
    feature_dim: int = 134,
    num_classes: int = 40,
    dropout_rate: float = 0.25,
    dense_dropout: float = 0.35,
    name: str = "compact_tcn"
) -> keras.Model:
    """Instantiate Compact TCN with configurable input feature dimension."""
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    
    # Block 1
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1d_entry")(inputs)
    x = layers.BatchNormalization(name="bn_entry")(x)
    x = layers.Dropout(dropout_rate, name="dropout_entry")(x)
    
    # Block 2 - Dilated residual bottleneck
    residual = x
    x = layers.Conv1D(64, kernel_size=3, padding="same", dilation_rate=2, activation="relu", name="conv1d_dilated_1")(x)
    x = layers.BatchNormalization(name="bn_res_1")(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1d_dilated_2")(x)
    x = layers.BatchNormalization(name="bn_res_2")(x)
    x = layers.add([x, residual], name="residual_add")
    x = layers.Dropout(dropout_rate, name="dropout_res")(x)
    
    # Block 3 - Deep dilated
    x = layers.Conv1D(128, kernel_size=3, padding="same", dilation_rate=4, activation="relu", name="conv1d_deep")(x)
    x = layers.BatchNormalization(name="bn_deep")(x)
    x = layers.Dropout(dropout_rate, name="dropout_deep")(x)
    
    # Classifier Head
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_projection")(x)
    x = layers.Dropout(dense_dropout, name="dropout_classifier")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def measure_cpu_latency(model: keras.Model, sample: np.ndarray, num_runs: int = 50) -> float:
    """Measure average single-sample CPU inference latency in milliseconds."""
    input_tensor = np.expand_dims(sample, axis=0).astype(np.float32)
    for _ in range(10):
        _ = model(input_tensor, training=False)
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(input_tensor, training=False)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return float(np.mean(timings))


def evaluate_on_validation(
    model: keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 40
) -> Dict[str, float]:
    """Compute validation metrics with zero_division=0."""
    val_probs = model.predict(X_val, batch_size=32, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    
    top1 = float(accuracy_score(y_val, val_preds))
    top5_preds = np.argsort(val_probs, axis=1)[:, -5:]
    top5 = float(np.mean([y_val[i] in top5_preds[i] for i in range(len(y_val))]))
    
    all_labels = list(range(num_classes))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, val_preds, labels=all_labels, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_val, val_preds, labels=all_labels, average="weighted", zero_division=0
    )
    
    return {
        "val_top1": top1,
        "val_top5": top5,
        "val_macro_f1": float(f1_macro),
        "val_weighted_f1": float(f1_wt),
        "val_macro_precision": float(p_macro),
        "val_macro_recall": float(r_macro)
    }


def run_single_experiment(
    exp_id: str,
    exp_name: str,
    use_velocity: bool,
    use_augmentation: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_134: np.ndarray,
    y_val: np.ndarray,
    ckpt_dir: Path,
    epochs: int = 80,
    batch_size: int = 32,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute one training run under strict experimental control."""
    set_random_seed(seed)
    print(f"\n=======================================================")
    print(f"  RUNNING {exp_id}: {exp_name}")
    print(f"  Velocity: {use_velocity} | Augmentation: {use_augmentation}")
    print(f"=======================================================")
    
    feat_dim = 263 if use_velocity else 134
    
    # Prepare Validation Data (deterministic, never augmented)
    if use_velocity:
        X_val = batch_compute_velocity(X_val_134)
    else:
        X_val = X_val_134
        
    # Prepare Model
    model = build_compact_tcn(
        sequence_length=64,
        feature_dim=feat_dim,
        num_classes=40,
        name=f"compact_tcn_{exp_id.lower()}"
    )
    param_count = model.count_params()
    print(f"Model parameters: {param_count:,} (Feature dim: {feat_dim})")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")]
    )
    
    ckpt_path = ckpt_dir / f"best_model_{exp_id.lower()}.keras"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1)
    ]
    
    # Dataset Preparation
    train_gen = DynamicAugmentedSequence(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        augment=use_augmentation,
        use_velocity=use_velocity,
        shuffle=True
    )
    
    t_start = time.perf_counter()
    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )
    t_train = time.perf_counter() - t_start
    
    # Load and evaluate best restored checkpoint on validation
    best_model = keras.models.load_model(str(ckpt_path))
    val_res = evaluate_on_validation(best_model, X_val, y_val, num_classes=40)
    
    hist_df = pd.DataFrame(history.history)
    best_ep = int(hist_df["val_loss"].idxmin() + 1)
    train_loss = float(hist_df.loc[best_ep - 1, "loss"])
    train_acc = float(hist_df.loc[best_ep - 1, "accuracy"])
    
    # File size and latency
    model_size_mb = os.path.getsize(str(ckpt_path)) / (1024 * 1024)
    latency_ms = measure_cpu_latency(best_model, X_val[0])
    
    print(f"\n--- {exp_id} Validation Results (Best Epoch {best_ep}) ---")
    print(f"  Val Top-1 Acc: {val_res['val_top1']*100:.2f}%")
    print(f"  Val Top-5 Acc: {val_res['val_top5']*100:.2f}%")
    print(f"  Val Macro F1:  {val_res['val_macro_f1']:.4f}")
    print(f"  Train Acc:     {train_acc*100:.2f}% (Loss: {train_loss:.4f})")
    print(f"  Latency:       {latency_ms:.2f} ms | Size: {model_size_mb:.2f} MB")
    
    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "use_velocity": use_velocity,
        "use_augmentation": use_augmentation,
        "feature_dim": feat_dim,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "latency_ms": latency_ms,
        "best_epoch": best_ep,
        "total_epochs": len(hist_df),
        "train_time_sec": t_train,
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "val_top1": val_res["val_top1"],
        "val_top5": val_res["val_top5"],
        "val_macro_f1": val_res["val_macro_f1"],
        "val_weighted_f1": val_res["val_weighted_f1"],
        "checkpoint_path": str(ckpt_path)
    }


def run_all_experiments(
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports",
    ckpt_dir: str = "models/checkpoints"
):
    """Execute complete 4-experiment suite and select best model on validation."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpts = Path(ckpt_dir)
    ckpts.mkdir(parents=True, exist_ok=True)
    
    print("=== EchoGesture Phase 5A: Accuracy Improvement Experiments ===")
    
    # 1. Load Preprocessed Partitions (134-D)
    train_npz = np.load(os.path.join(data_dir, "train_sequences.npz"))
    val_npz = np.load(os.path.join(data_dir, "val_sequences.npz"))
    
    X_train_134 = train_npz["X"]
    y_train = train_npz["y"]
    X_val_134 = val_npz["X"]
    y_val = val_npz["y"]
    
    experiments_def = [
        ("EXP_A", "Baseline Compact TCN (134-D, No Aug)", False, False),
        ("EXP_B", "Compact TCN + Velocity (263-D, No Aug)", True, False),
        ("EXP_C", "Compact TCN + Dynamic Augmentation (134-D)", False, True),
        ("EXP_D", "Compact TCN + Velocity + Dynamic Augmentation (263-D)", True, True),
    ]
    
    results = []
    for exp_id, exp_name, use_vel, use_aug in experiments_def:
        res = run_single_experiment(
            exp_id=exp_id,
            exp_name=exp_name,
            use_velocity=use_vel,
            use_augmentation=use_aug,
            X_train=X_train_134,
            y_train=y_train,
            X_val_134=X_val_134,
            y_val=y_val,
            ckpt_dir=ckpts,
            epochs=80,
            batch_size=32,
            seed=42
        )
        results.append(res)
        
    # Save Experiments Summary to CSV
    res_df = pd.DataFrame(results)
    csv_path = out_path / "phase5a_experiments_history.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved experiments history to {csv_path}")
    
    # Display Validation Comparison Table
    print("\n=========================================================================================")
    print("                      PHASE 5A VALIDATION COMPARISON & MODEL SELECTION                    ")
    print("=========================================================================================")
    print(res_df[[
        "exp_id", "feature_dim", "param_count", "val_top1", "val_top5", "val_macro_f1", "train_accuracy", "latency_ms"
    ]].to_string(index=False))
    
    # Model Selection Rule: Highest Validation Macro F1, breaking ties with Val Top-1
    res_df = res_df.sort_values(by=["val_macro_f1", "val_top1"], ascending=[False, False])
    best_row = res_df.iloc[0]
    
    print("\n=========================================================================================")
    print(f" WINNING CONFIGURATION: {best_row['exp_id']} ({best_row['exp_name']})")
    print(f" Val Macro F1: {best_row['val_macro_f1']:.4f} | Val Top-1: {best_row['val_top1']*100:.2f}% | Val Top-5: {best_row['val_top5']*100:.2f}%")
    print("=========================================================================================")
    
    # Copy best checkpoint to designated best improved model path
    best_ckpt_src = best_row["checkpoint_path"]
    best_ckpt_dst = ckpts / "best_improved_model_v5a.keras"
    shutil.copyfile(best_ckpt_src, str(best_ckpt_dst))
    print(f"Copied winning model to {best_ckpt_dst}")
    
    # Save selection metadata
    sel_path = out_path / "phase5a_winning_selection.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(best_row.to_dict(), f, indent=2)
    print(f"Saved winning model metadata to {sel_path}")
    
    return best_row.to_dict(), res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Phase 5A Experiments")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()
    
    run_all_experiments(args.data_dir, args.output_dir, args.ckpt_dir)
