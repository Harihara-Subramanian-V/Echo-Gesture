# EchoGesture Phase 5A: Recognition Accuracy Improvement Report

**Project**: EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Date**: November 20, 2026  
**Pipeline Step**: Phase 5A - Accuracy Improvement via Velocity Features & Dynamic Data Augmentation  
**Evaluation Standard**: Controlled Ablation Study on Validation Set $\to$ Single Unbiased Evaluation on Held-Out Test Set  
**Status**: COMPLETE (Significant Measurable Improvement Demonstrated)  

---

## 1. Executive Summary

Phase 5A systematically investigated two targeted enhancements to resolve the performance bottlenecks identified in Phase 4:
1. **First-order temporal velocity features** ($\Delta \mathbf{x} / \Delta t$) with transition spike prevention.
2. **Dynamic training-only kinematic data augmentation** (smooth temporal speed warping $0.85\times - 1.15\times$, spatial scaling $\pm 10\%$, and sensor jitter).

Across 4 strictly controlled ablation experiments evaluated under identical signer-independent conditions, the combination of velocity features and dynamic augmentation (**EXP_D**) emerged as the decisive winner on validation data.

When evaluated strictly once on the held-out test partition (197 clips across 10 unseen signers), the winning model demonstrated substantial, measurable improvements across every metric:

```
+---------------------------------------------------------------------------------------------------+
|                         FINAL HELD-OUT TEST PERFORMANCE: BASELINE VS. PHASE 5A                    |
+------------------------------------+--------------------+--------------------+--------------------+
| Metric                             | Phase 4 Baseline   | Phase 5A (EXP_D)   | Absolute Delta     |
+------------------------------------+--------------------+--------------------+--------------------+
| Top-1 Accuracy                     | 52.79% (104/197)   | 64.47% (127/197)   | +11.68%            |
| Top-5 Accuracy                     | 82.23% (162/197)   | 84.77% (167/197)   | +2.54%             |
| Macro F1-Score                     | 0.4700             | 0.6287             | +0.1587            |
| Weighted F1-Score                  | 0.4888             | 0.6385             | +0.1497            |
| Macro Recall                       | 0.5050             | 0.6554             | +0.1504            |
| Signs with 0% Recall               | 7 / 40 signs       | 1 / 40 signs       | -6 signs recovered |
| Signs with 100% Recall             | 8 / 40 signs       | 9 / 40 signs       | +1 sign            |
+------------------------------------+--------------------+--------------------+--------------------+
| Parameter Count                    | 87,336             | 112,104            | +24,768 (< 150k)   |
| Model File Size (.keras)           | 1.07 MB            | 1.36 MB            | +0.29 MB (< 20 MB) |
| CPU Inference Latency (Mean)       | 17.57 ms           | 21.79 ms           | +4.22 ms (< 30 ms) |
| CPU Inference Latency (Median)     | 16.96 ms           | 20.89 ms           | +3.93 ms (< 30 ms) |
+------------------------------------+--------------------+--------------------+--------------------+
```

---

## 2. Design of First-Order Temporal Velocity Features

### 2.1 Theoretical Rationale
In Phase 4, the primary misclassifications occurred on dynamic flick gestures (`go`, `many`) and lifting motions (`help`). In static coordinate space, a hand stationary in front of the chest and a hand rapidly accelerating forward produce nearly identical normalized spatial frames. Temporal velocity directly captures rate of displacement, differentiating dynamic gesture impulses from static hand shapes.

### 2.2 Transition Spike Prevention
A critical danger when computing velocities on landmark streams with intermittent detections is **transition spiking**:
- If a hand appears at frame $t$ ($\mathbf{c}_t \ne \mathbf{0}$) having been absent at $t-1$ ($\mathbf{c}_{t-1} = \mathbf{0}$), a naive difference $\mathbf{c}_t - \mathbf{0} = \mathbf{c}_t$ registers a massive artificial jump from the top-left screen origin $(0,0,0)$ to the hand's appearance coordinate.
- Conversely, when a hand disappears at $t$, $\mathbf{0} - \mathbf{c}_{t-1}$ produces an artificial negative velocity spike.

To eliminate this artifact, velocity is conditioned on simultaneous presence:
$$\mathbf{v}_L(t) = \begin{cases} \mathbf{c}_L(t) - \mathbf{c}_L(t-1), & \text{if } p_L(t) == 1.0 \land p_L(t-1) == 1.0 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
$$\mathbf{v}_R(t) = \begin{cases} \mathbf{c}_R(t) - \mathbf{c}_R(t-1), & \text{if } p_R(t) == 1.0 \land p_R(t-1) == 1.0 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
$$\mathbf{v}_{\text{inter}}(t) = \begin{cases} \mathbf{f}_{\text{inter}}(t) - \mathbf{f}_{\text{inter}}(t-1), & \text{if } p_{\text{both}}(t) == 1.0 \land p_{\text{both}}(t-1) == 1.0 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
Boundary condition at sequence start ($t=0$): $\mathbf{v}(0) = \mathbf{0}$.

### 2.3 Feature Dimensionality Accounting
- Base Landmark Features: **134 dimensions**
  - Left Hand Landmarks: 63 ($21 \times 3$)
  - Right Hand Landmarks: 63 ($21 \times 3$)
  - Presence Flags: 2 (Left, Right)
  - Handedness Confidence Scores: 2
  - Co-Presence Flag: 1
  - Interaction Features: 3 (Euclidean distance, horizontal offset $\Delta x$, vertical offset $\Delta y$)
- Temporal Velocity Features: **129 dimensions**
  - Left Hand Landmark Velocities: 63
  - Right Hand Landmark Velocities: 63
  - Interaction Velocities: 3 ($\Delta d, \Delta(\Delta x), \Delta(\Delta y)$)
- **Total Augmented Feature Dimension**: **263 dimensions**.

---

## 3. Dynamic Training-Only Data Augmentation

To address signer variance (the ~18% generalization gap between training signers and unseen test signers), a dynamic on-the-fly augmentation pipeline was implemented in `scripts/augment_sequences.py`:

1. **Smooth Temporal Speed Warping ($0.85\times - 1.15\times$)**:
   - Randomly samples continuous temporal query windows, stretching or compressing gesture dynamics while preserving the fixed 64-frame output shape.
   - Presence flags are resampled via nearest-neighbor, and continuous landmark coordinates are interpolated along active trajectory lines.
2. **Anatomical Spatial Scaling ($\pm 10\%$)**:
   - Scales active hand coordinates by $\gamma \sim \text{Uniform}(0.90, 1.10)$ relative to the wrist anchor.
   - Simulates diverse hand sizes, signer body proportions, and varying camera distances without altering valid joint topology.
3. **Sensor Jitter Perturbation**:
   - Adds zero-mean Gaussian noise ($\sigma = 0.005$) strictly to active detected frames, simulating sensor noise and edge camera shake.
4. **Strict Isolation & Zero Data Leakage**:
   - Implemented via a custom Keras `PyDataset` (`DynamicAugmentedSequence`).
   - Augmentation operates on-the-fly during mini-batch sampling.
   - Disk archives, validation sequences, and test sequences were **never modified or augmented**.

---

## 4. Controlled Ablation Experiments & Model Selection

Four configurations were trained under identical conditions:
- **Optimizer**: Adam ($\eta_0 = 10^{-3}$, $\text{clipnorm}=1.0$)
- **Scheduler**: `ReduceLROnPlateau` ($\text{factor}=0.5, \text{patience}=6$)
- **Early Stopping**: $\text{patience}=15$ monitored on `val_loss`, restoring best weights
- **Batch Size**: 32 | **Random Seed**: 42

```
+-----------------------------------------------------------------------------------------------------------------+
|                                 PHASE 5A VALIDATION ABLATION COMPARISON                                         |
+-------+--------------------------------------+--------+---------+----------+----------+--------------+----------+
| Exp ID| Configuration                        | Feat D | Params  | Val Top-1| Val Top-5| Val Macro F1 | Latency  |
+-------+--------------------------------------+--------+---------+----------+----------+--------------+----------+
| EXP_A | Baseline Compact TCN (No Aug)        | 134    | 87,336  | 57.63%   | 83.97%   | 0.5168       | 17.86 ms |
| EXP_B | Compact TCN + Velocity (No Aug)      | 263    | 112,104 | 61.07%   | 82.82%   | 0.5530       | 17.15 ms |
| EXP_C | Compact TCN + Dynamic Aug (134-D)    | 134    | 87,336  | 62.60%   | 83.59%   | 0.5718       | 18.59 ms |
| EXP_D | Compact TCN + Velocity + Aug (263-D) | 263    | 112,104 | 65.27%   | 86.64%   | 0.5833       | 16.99 ms |
+-------+--------------------------------------+--------+---------+----------+----------+--------------+----------+
```

### Analysis of Validation Results:
- **Adding Velocity Alone (EXP_B vs EXP_A)**: Boosted Val Top-1 from 57.63% to 61.07% (+3.44%) and Val Macro F1 from 0.5168 to 0.5530 (+0.0362).
- **Adding Augmentation Alone (EXP_C vs EXP_A)**: Boosted Val Top-1 to 62.60% (+4.97%) and Val Macro F1 to 0.5718 (+0.0550).
- **Synergistic Combination (EXP_D)**: Achieved the highest performance across all validation metrics: **Val Top-1: 65.27%**, **Val Top-5: 86.64%**, and **Val Macro F1: 0.5833**.

**Decision**: **EXP_D** was selected strictly based on validation performance and saved to [best_improved_model_v5a.keras](file:///c:/EchoCam/models/checkpoints/best_improved_model_v5a.keras).

---

## 5. Unbiased Evaluation on Held-Out Test Set (Unseen Signers)

Evaluating EXP_D strictly once on the held-out test split (197 clips, 40 classes):

### 5.1 Global Metrics Comparison
- **Top-1 Accuracy**: **64.47%** (+11.68% over Baseline 52.79%)
- **Top-5 Accuracy**: **84.77%** (+2.54% over Baseline 82.23%)
- **Macro F1-Score**: **0.6287** (+0.1587 over Baseline 0.4700)
- **Weighted F1-Score**: **0.6385** (+0.1497 over Baseline 0.4888)

### 5.2 Resolution of Zero-Recall Sign Classes
In the Phase 4 baseline, 7 sign classes suffered from 0% recall. Under EXP_D, **6 of these 7 signs were successfully recovered**:

| Sign Class | Test Support | Baseline Recall | Phase 5A Recall | Baseline F1 | Phase 5A F1 | Absolute $\Delta$ F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **help** | 3 | 0.0% | **66.7%** | 0.000 | **0.800** | **+0.800** |
| **go** | 7 | 0.0% | **57.1%** | 0.000 | **0.667** | **+0.667** |
| **many** | 5 | 0.0% | **60.0%** | 0.000 | **0.600** | **+0.600** |
| **thin** | 3 | 0.0% | **33.3%** | 0.000 | **0.500** | **+0.500** |
| **bed** | 7 | 0.0% | **14.3%** | 0.000 | **0.222** | **+0.222** |
| **woman** | 5 | 0.0% | **20.0%** | 0.000 | **0.182** | **+0.182** |
| **bowling**| 2 | 0.0% | 0.0% | 0.000 | 0.000 | 0.000 (Low support) |

### 5.3 Top 10 Overall Performing Signs in Phase 5A:
| Sign Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **black** | 3 | 1.000 | 1.000 | **1.000** |
| **drink** | 1 | 1.000 | 1.000 | **1.000** |
| **family** | 5 | 1.000 | 1.000 | **1.000** |
| **graduate** | 2 | 1.000 | 1.000 | **1.000** |
| **table** | 5 | 1.000 | 1.000 | **1.000** |
| **now** | 10 | 0.833 | 1.000 | **0.909** |
| **thanksgiving**| 3 | 0.750 | 1.000 | **0.857** |
| **computer** | 2 | 0.667 | 1.000 | **0.800** |
| **help** | 3 | 1.000 | 0.667 | **0.800** |
| **yes** | 3 | 1.000 | 0.667 | **0.800** |

- Per-class metrics table: [per_class_v5a_metrics.csv](file:///c:/EchoCam/reports/per_class_v5a_metrics.csv)
- Confusion matrix visualization: [confusion_matrix_v5a.png](file:///c:/EchoCam/reports/figures/confusion_matrix_v5a.png)
- Structured evaluation metrics: [phase5a_final_test_evaluation.json](file:///c:/EchoCam/reports/phase5a_final_test_evaluation.json)

---

## 6. Edge Constraint Verification & Resource Profiling

- **Parameter Budget**: 112,104 parameters (budget: < 150,000) $\to$ **25.3% under maximum parameter budget**.
- **Model Storage**: 1.36 MB (`.keras`), ~450 KB (`.tflite` float32) (budget: < 20 MB) $\to$ **97.7% under budget**.
- **CPU Inference Latency (Single-Sample 100-Trial Benchmark)**:
  - Mean: **21.79 ms**
  - Median: **20.89 ms**
  - 95th Percentile: **26.96 ms**
  - Min / Max: 18.04 ms / 32.18 ms
  - Target constraint (< 30 ms): **Fully met with 27.4% latency headroom**.
- **Architecture Integrity**: Built strictly with standard 1D convolutions, Batch Normalization, Add, ReLU, and GlobalAveragePooling1D. Fully ready for clean TFLite export and INT8 post-training quantization.

---

## 7. Deliverables & Preserved Artifacts

1. **Feature Engineering & Augmentation Module**: [augment_sequences.py](file:///c:/EchoCam/scripts/augment_sequences.py)
2. **Ablation Training Script**: [train_accuracy_experiments.py](file:///c:/EchoCam/scripts/train_accuracy_experiments.py)
3. **Final Test Evaluation Script**: [evaluate_accuracy_experiments.py](file:///c:/EchoCam/scripts/evaluate_accuracy_experiments.py)
4. **Configuration (Non-destructive)**: [model_config_v5a.yaml](file:///c:/EchoCam/configs/model_config_v5a.yaml)
5. **Winning Checkpoint**: [best_improved_model_v5a.keras](file:///c:/EchoCam/models/checkpoints/best_improved_model_v5a.keras)
6. **Ablation History Log**: [phase5a_experiments_history.csv](file:///c:/EchoCam/reports/phase5a_experiments_history.csv)
7. **Per-Class Metrics CSV**: [per_class_v5a_metrics.csv](file:///c:/EchoCam/reports/per_class_v5a_metrics.csv)
8. **Test Confusion Matrix**: [confusion_matrix_v5a.png](file:///c:/EchoCam/reports/figures/confusion_matrix_v5a.png)
9. **Original Phase 4 Artifacts**: All strictly preserved untouched ([best_baseline_model.keras](file:///c:/EchoCam/models/checkpoints/best_baseline_model.keras), [baseline_training_report.md](file:///c:/EchoCam/reports/baseline_training_report.md)).

---

## 8. Conclusion & Stop Condition

Phase 5A has concluded. Recognition accuracy has been improved on the held-out test split from **52.79% to 64.47% Top-1** and **0.4700 to 0.6287 Macro F1**, recovering 6 of 7 previously failing sign classes while maintaining real-time latency (21.79 ms) and compact edge model size (1.36 MB).

In accordance with strict instructions, execution is stopped before quantization, deployment, or application integration pending user approval.
