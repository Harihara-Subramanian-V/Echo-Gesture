"""
EchoGesture Phase 4: Baseline Model Training Pipeline
Trains compact sequence models (Compact TCN / Compact GRU) on preprocessed landmark sequences.

Target Constraints:
- Parameter count < 150k
- Pure TFLite built-in compatibility
- < 20 MB model footprint
- < 30 ms inference latency on CPU
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, callbacks, optimizers, losses, metrics


def set_random_seed(seed: int = 42):
    """Ensure strict deterministic reproducibility."""
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
    dense_dropout: float = 0.35
) -> keras.Model:
    """
    Build Compact Temporal Convolutional Network (1D-CNN) with dilated residual connections.
    
    Architecture is fully compatible with TFLite builtins, avoiding dynamic RNN lowering issues.
    """
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    
    # Block 1: Initial temporal feature expansion
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1d_entry")(inputs)
    x = layers.BatchNormalization(name="bn_entry")(x)
    x = layers.Dropout(dropout_rate, name="dropout_entry")(x)
    
    # Block 2: Dilated residual bottleneck (receptive field = 5 frames)
    residual = x
    x = layers.Conv1D(64, kernel_size=3, padding="same", dilation_rate=2, activation="relu", name="conv1d_dilated_1")(x)
    x = layers.BatchNormalization(name="bn_res_1")(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1d_dilated_2")(x)
    x = layers.BatchNormalization(name="bn_res_2")(x)
    x = layers.add([x, residual], name="residual_add")
    x = layers.Dropout(dropout_rate, name="dropout_res")(x)
    
    # Block 3: Higher-capacity temporal block with dilation 4 (receptive field = 13 frames)
    x = layers.Conv1D(128, kernel_size=3, padding="same", dilation_rate=4, activation="relu", name="conv1d_deep")(x)
    x = layers.BatchNormalization(name="bn_deep")(x)
    x = layers.Dropout(dropout_rate, name="dropout_deep")(x)
    
    # Aggregation & Classification Head
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_projection")(x)
    x = layers.Dropout(dense_dropout, name="dropout_classifier")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="compact_tcn")


def build_compact_gru(
    sequence_length: int = 64,
    feature_dim: int = 134,
    num_classes: int = 40,
    dropout_rate: float = 0.25,
    dense_dropout: float = 0.35
) -> keras.Model:
    """
    Build Compact Unrolled GRU model for recurrent temporal sequence comparison.
    """
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    x = layers.GRU(64, return_sequences=False, unroll=True, dropout=dropout_rate, name="gru_unrolled")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_projection")(x)
    x = layers.Dropout(dense_dropout, name="dropout_classifier")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="compact_gru")


def load_dataset(preprocessed_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed sequences and integer targets for train and val splits."""
    p = Path(preprocessed_dir)
    train_data = np.load(p / "train_sequences.npz")
    val_data = np.load(p / "val_sequences.npz")
    
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_val = val_data["X"]
    y_val = val_data["y"]
    
    return X_train, y_train, X_val, y_val


def plot_training_curves(history_df: pd.DataFrame, output_path: str):
    """Plot Loss and Accuracy curves for training and validation."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss subplot
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Accuracy subplot
    axes[1].plot(history_df["epoch"], history_df["accuracy"] * 100, label="Train Top-1 Acc", color="#1f77b4", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"] * 100, label="Val Top-1 Acc", color="#ff7f0e", linewidth=2)
    if "top_5_accuracy" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["top_5_accuracy"] * 100, label="Train Top-5 Acc", color="#1f77b4", linestyle=":")
        axes[1].plot(history_df["epoch"], history_df["val_top_5_accuracy"] * 100, label="Val Top-5 Acc", color="#ff7f0e", linestyle=":")
    axes[1].set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved training history curves to {output_path}")


def train(config_path: str = "configs/model_config.yaml", arch_override: str = None):
    """Main training execution routine."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    set_random_seed(cfg["experiment"]["random_seed"])
    
    arch = arch_override or cfg["architecture"]["selected"]
    print(f"\n=======================================================")
    print(f"  EchoGesture Training: {arch.upper()}")
    print(f"=======================================================")
    
    seq_len = cfg["dataset"]["sequence_length"]
    feat_dim = cfg["dataset"]["feature_dim"]
    num_classes = cfg["dataset"]["num_classes"]
    
    # 1. Load Data
    data_dir = cfg["dataset"]["preprocessed_dir"]
    X_train, y_train, X_val, y_val = load_dataset(data_dir)
    print(f"Dataset Loaded from {data_dir}:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    
    # 2. Build Model
    if arch == "compact_tcn":
        tcn_cfg = cfg["architecture"]["compact_tcn"]
        model = build_compact_tcn(
            sequence_length=seq_len,
            feature_dim=feat_dim,
            num_classes=num_classes,
            dropout_rate=tcn_cfg["dropout"],
            dense_dropout=tcn_cfg["dense_dropout"]
        )
    elif arch == "compact_gru":
        gru_cfg = cfg["architecture"]["compact_gru"]
        model = build_compact_gru(
            sequence_length=seq_len,
            feature_dim=feat_dim,
            num_classes=num_classes,
            dropout_rate=gru_cfg["dropout"],
            dense_dropout=gru_cfg["dense_dropout"]
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    model.summary()
    total_params = model.count_params()
    max_budget = cfg["edge_deployment_constraints"]["max_parameters"]
    print(f"\nParameter Count: {total_params:,} (Budget: {max_budget:,})")
    assert total_params <= max_budget, f"Model parameter count ({total_params}) exceeds budget ({max_budget})!"
    
    # 3. Compile Model
    lr = cfg["training"]["learning_rate"]
    clipnorm = cfg["training"]["clipnorm"]
    optimizer = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    loss_fn = losses.SparseCategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    
    # 4. Callbacks Setup
    ckpt_dir = Path(cfg["checkpoints"]["directory"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = cfg["checkpoints"]["best_model_path"]
    
    es_cfg = cfg["training"]["early_stopping"]
    lr_cfg = cfg["training"]["lr_reduction"]
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            restore_best_weights=es_cfg["restore_best_weights"],
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=lr_cfg["monitor"],
            factor=lr_cfg["factor"],
            patience=lr_cfg["patience"],
            min_lr=lr_cfg["min_lr"],
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=es_cfg["monitor"],
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 5. Execute Training
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    print(f"\nStarting training for {epochs} epochs (batch_size={batch_size})...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback_list,
        shuffle=True,
        verbose=1
    )
    
    # 6. Save History & Curves
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = range(1, len(history_df) + 1)
    
    hist_csv = cfg["checkpoints"]["history_csv_path"]
    Path(hist_csv).parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(hist_csv, index=False)
    print(f"\nSaved training history log to {hist_csv}")
    
    hist_plot = cfg["checkpoints"]["history_plot_path"]
    plot_training_curves(history_df, hist_plot)
    
    # 7. Print Best Validation Results
    best_epoch_idx = history_df["val_loss"].idxmin()
    best_epoch = history_df.iloc[best_epoch_idx]
    print(f"\n--- Best Validation Epoch: {int(best_epoch['epoch'])} ---")
    print(f"  Val Loss:      {best_epoch['val_loss']:.4f}")
    print(f"  Val Top-1 Acc: {best_epoch['val_accuracy']*100:.2f}%")
    print(f"  Val Top-5 Acc: {best_epoch['val_top_5_accuracy']*100:.2f}%")
    print(f"  Train Loss:    {best_epoch['loss']:.4f}")
    print(f"  Train Top-1:   {best_epoch['accuracy']*100:.2f}%")
    print(f"  Model Saved:   {best_model_path}")
    
    return history_df, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Baseline Training")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--arch", type=str, default=None)
    args = parser.parse_args()
    
    train(args.config, args.arch)
