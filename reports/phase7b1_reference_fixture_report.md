# EchoGesture Phase 7B.1: Python Reference Fixture Generation Report
**Track 2: Samsung XR Hackathon 2026**  
**Component:** Reference Fixture Pipeline, Numerical Ground Truth & Deterministic Parity Artifacts  
**Execution Strategy:** Parallel Specialist Swarm (3 Specialists + 1 Coordinating Main Agent)  
**Evaluator & Execution Engine:** Gemini 3.8 Flash (High) ONLY  
**Date:** 2026-11-20  
**Workspace:** `C:\EchoCam`  

---

## 1. Executive Summary

Phase 7B.1 establishes the deterministic numerical ground truth for the native Android Kotlin port of **EchoGesture**. To guarantee that the upcoming Kotlin recognition pipeline produces identical outputs to the validated Python system, comprehensive reference fixtures were generated across all three core pipeline stages directly from the frozen Python implementation.

The generation was coordinated by the Main Agent and executed using three independent specialist subagents with strict file separation:
1. **Agent 1 (Feature Pipeline Specialist):** Authored `generate_stage1_feature_fixtures.py` and generated canonical 134-D feature representations across 7 diverse spatial, hand-presence, and slot-tracking configurations.
2. **Agent 2 (Temporal Preprocessing Specialist):** Authored `generate_stage2_temporal_fixtures.py` and generated canonical 263-D sequence tensors (64 frames) across 8 temporal length, resampling, zero-clamping, and velocity-transition configurations.
3. **Agent 3 (INT8 Inference Specialist):** Authored `generate_stage3_int8_fixtures.py` and generated bit-exact quantized inputs, raw INT8 logits, dequantized probabilities, monotonic argmax predictions, and top-5 rankings across 8 synthetic and held-out test configurations.

### Key Milestone Achievements:
- **Bit-for-Bit Determinism Verified:** All fixtures were regenerated across two independent sequential runs; binary SHA-256 checksums were verified 100% identical.
- **Zero Production Mutation:** Zero existing Python production scripts, models, dataset splits, or scaler files were altered.
- **Zero Data Leakage:** Scaler statistics originate exclusively from the training partition (`data/processed/sequences_v3/scaler_params.npz`).
- **Comprehensive Automated Validation:** A master automated test suite ([tests/test_parity_fixtures.py](file:///c:/EchoCam/tests/test_parity_fixtures.py)) alongside stage-specific test suites validates all 23 fixture cases with zero failures (75/75 total repository tests passing).

---

## 2. Environment & Dependency Telemetry

All reference fixtures were generated inside the project `.venv` runtime:

| Component | Verified Version | Verification Command / Source |
| :--- | :--- | :--- |
| **Python** | **3.10.11** (64-bit AMD64) | `C:\EchoCam\.venv\Scripts\python.exe` |
| **NumPy** | **1.26.4** | `numpy.__version__` |
| **MediaPipe** | **0.10.21** | `mediapipe.__version__` |
| **TensorFlow** | **2.16.2** | `tensorflow.__version__` |
| **OpenCV** | **4.10.0.84** | `cv2.__version__` |
| **Model Checkpoint** | `models/quantized/echogesture_exp_d_int8.tflite` | **135,464 bytes** |
| **Model SHA-256** | `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` | **Bit-for-bit exact match** |
| **Training Scaler** | `data/processed/sequences_v3/scaler_params.npz` | **`mean` & `std` shape (134,)** |

---

## 3. Specialist Agent Execution & Artifact Overview

### 3.1. Agent 1: Feature Pipeline Reference (134-D)
- **Specialist Lead:** Agent 1 (Feature Pipeline Specialist)
- **Generator Script:** [tests/fixtures/android_parity/generate_stage1_feature_fixtures.py](file:///c:/EchoCam/tests/fixtures/android_parity/generate_stage1_feature_fixtures.py)
- **Binary Fixture:** [tests/fixtures/android_parity/stage1_features.npz](file:///c:/EchoCam/tests/fixtures/android_parity/stage1_features.npz) (5,834 bytes)
- **JSON Fixture:** [tests/fixtures/android_parity/stage1_features.json](file:///c:/EchoCam/tests/fixtures/android_parity/stage1_features.json) (143,730 bytes)
- **Unit Tests:** [tests/test_stage1_parity_fixtures.py](file:///c:/EchoCam/tests/test_stage1_parity_fixtures.py) (12 passing tests)

#### 134-D Dimension Layout Specification:
- **`[0:63]`**: Slot 0 (Viewer-Left Hand) 21 normalized landmarks $(x, y, z)$. Strictly $0.0$ when left hand is absent.
- **`[63:126]`**: Slot 1 (Viewer-Right Hand) 21 normalized landmarks $(x, y, z)$. Strictly $0.0$ when right hand is absent.
- **`[126]`**: Left Hand Presence flag ($1.0$ if present, $0.0$ if absent).
- **`[127]`**: Right Hand Presence flag ($1.0$ if present, $0.0$ if absent).
- **`[128]`**: Left Hand classification confidence score ($[0.0, 1.0]$, $0.0$ if absent).
- **`[129]`**: Right Hand classification confidence score ($[0.0, 1.0]$, $0.0$ if absent).
- **`[130]`**: Both Hands Co-presence flag ($1.0$ if both present, $0.0$ otherwise).
- **`[131]`**: Aspect-ratio corrected diagonal-normalized inter-wrist Euclidean distance:
  $$d_{\text{geom}} = \frac{\sqrt{(\Delta x_{\text{norm}} \cdot W_{\text{px}})^2 + (\Delta y_{\text{norm}} \cdot H_{\text{px}})^2}}{\sqrt{W_{\text{px}}^2 + H_{\text{px}}^2}}$$
- **`[132:134]`**: Aspect-ratio corrected diagonal-normalized inter-wrist offsets $(\Delta x_{\text{px}} / D_{\text{diag}}, \Delta y_{\text{px}} / D_{\text{diag}})$.

#### Test Case Matrix:
| Case Identifier | Scenario / Description | Key Invariant Verified |
| :--- | :--- | :--- |
| `case_1_left_only` | Single Left hand detected (confidence 0.95); Right hand absent. | Right slot `[63:126]` strictly $0.0$; presence $p_L=1, p_R=0, p_{\text{both}}=0$; geometry $0.0$. |
| `case_2_right_only` | Single Right hand detected (confidence 0.92); Left hand absent. | Left slot `[0:63]` strictly $0.0$; presence $p_L=0, p_R=1, p_{\text{both}}=0$; geometry $0.0$. |
| `case_3_both_hands` | Both hands detected in standard 640x480 frame (L wrist at $(0.3, 0.6)$, R wrist at $(0.7, 0.5)$). | $d_{\text{geom}} = 0.325576$, $dx_{\text{geom}} = 0.320000$, $dy_{\text{geom}} = -0.060000$; $p_{\text{both}} = 1.0$. |
| `case_4_missing_hands` | Blank frame; neither hand detected by detector. | All 134 vector elements are strictly $0.0$. |
| `case_5_transition_sequence` | 5-frame temporal sequence: Empty $\to$ Left enters $\to$ Right enters $\to$ Motion $\to$ Left leaves. | Validates slot continuity across rapid entry and exit transitions. |
| `case_6_hungarian_tracking` | Ambiguous duplicate 'Left' labels with inverted x-positions ($c_0$ at $0.74$, $c_1$ at $0.26$). | Hungarian cost minimization assigns $c_1 \to$ Slot 0 and $c_0 \to$ Slot 1, preserving trajectory. |
| `case_7_aspect_ratios` | Compares 640x480 (4:3) vs 1280x720 (16:9) vs 1920x1080 (16:9). | Proves identical geometric coordinates under proportional aspect ratios ($10^{-6}$ precision). |

---

### 3.2. Agent 2: Temporal Preprocessing Reference (263-D)
- **Specialist Lead:** Agent 2 (Temporal Preprocessing Specialist)
- **Generator Script:** [tests/fixtures/android_parity/generate_stage2_temporal_fixtures.py](file:///c:/EchoCam/tests/fixtures/android_parity/generate_stage2_temporal_fixtures.py)
- **Binary Fixture:** [tests/fixtures/android_parity/stage2_temporal.npz](file:///c:/EchoCam/tests/fixtures/android_parity/stage2_temporal.npz) (1,065.6 KB)
- **JSON Fixture:** [tests/fixtures/android_parity/stage2_temporal.json](file:///c:/EchoCam/tests/fixtures/android_parity/stage2_temporal.json) (9.84 MB)
- **Unit Tests:** [tests/test_stage2_temporal_fixtures.py](file:///c:/EchoCam/tests/test_stage2_temporal_fixtures.py) (5 passing tests)

#### Preprocessing & Transformation Invariants:
1. **Dynamic Grid Resampling:** Variable-length sequence $T \to 64$ frames using nearest-neighbor on binary presence flags and active-frame linear interpolation along trajectories.
2. **Active-Frame Z-Score Standardization:**
   $$x_{\text{std}} = \begin{cases} \frac{x - \mu}{\sigma + 10^{-6}} & \text{if hand is active in frame } t \\ 0.0 & \text{if hand is inactive (preserves strict zero-masking)} \end{cases}$$
3. **129-D Velocity Features with Transition Spike Prevention:**
   - Left Hand Velocity (63-D): $v_L[t] = x_L[t] - x_L[t-1]$ iff $p_L[t] \land p_L[t-1]$, else $0.0$.
   - Right Hand Velocity (63-D): $v_R[t] = x_R[t] - x_R[t-1]$ iff $p_R[t] \land p_R[t-1]$, else $0.0$.
   - Interaction Velocity (3-D): $v_{\text{inter}}[t] = x_{\text{inter}}[t] - x_{\text{inter}}[t-1]$ iff $p_{\text{both}}[t] \land p_{\text{both}}[t-1]$, else $0.0$.
   - Boundary condition: $v[0] = 0.0$ strictly enforced across all 129 channels.
4. **Concatenation:** $[x_{\text{std}}, v_L, v_R, v_{\text{inter}}] \in \mathbb{R}^{64 \times 263}$.

#### Test Case Matrix:
| Case Identifier | Raw Length $T$ | Resampled $T$ | Output Shape | Non-Zero % | Offline Diff | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `case_1_short_sequence` | 24 | 64 | $(64, 263)$ | 59.9% | $3.11 \times 10^{-4}$ | **PASS** |
| `case_2_full_sequence` | 64 | 64 | $(64, 263)$ | 62.2% | $3.11 \times 10^{-4}$ | **PASS** |
| `case_3_long_sequence` | 96 | 64 | $(64, 263)$ | 63.0% | $5.04 \times 10^{-4}$ | **PASS** |
| `case_4_sparse_detections` | 50 | 64 | $(64, 263)$ | 9.0% | $3.04 \times 10^{-4}$ | **PASS** |
| `case_5_single_hand_continuous` | 64 | 64 | $(64, 263)$ | 48.3% | $3.11 \times 10^{-4}$ | **PASS** |
| `case_6_two_hand_continuous` | 64 | 64 | $(64, 263)$ | 99.2% | $3.11 \times 10^{-4}$ | **PASS** |
| `case_7_alternating_hands` | 64 | 64 | $(64, 263)$ | 47.9% | $2.14 \times 10^{-4}$ | **PASS** |
| `case_8_idle_window` | 64 | 64 | $(64, 263)$ | 0.0% | $0.000000$ | **PASS** |

---

### 3.3. Agent 3: INT8 Inference Parity Reference ([1, 64, 263] -> [1, 40])
- **Specialist Lead:** Agent 3 (INT8 Inference Specialist)
- **Generator Script:** [tests/fixtures/android_parity/generate_stage3_int8_fixtures.py](file:///c:/EchoCam/tests/fixtures/android_parity/generate_stage3_int8_fixtures.py)
- **Binary Fixture:** [tests/fixtures/android_parity/stage3_inference.npz](file:///c:/EchoCam/tests/fixtures/android_parity/stage3_inference.npz) (682.8 KB)
- **JSON Fixture:** [tests/fixtures/android_parity/stage3_inference.json](file:///c:/EchoCam/tests/fixtures/android_parity/stage3_inference.json) (11.78 MB)
- **Unit Tests:** [tests/test_stage3_int8_fixtures.py](file:///c:/EchoCam/tests/test_stage3_int8_fixtures.py) (9 passing tests)

#### Exact Quantization & Inference Contract:
- **Input Tensor Contract:**
  - Shape: `[1, 64, 263]`, dtype `int8`
  - Scale: $S_{\text{in}} = 0.05794047564268112$
  - Zero Point: $Z_{\text{in}} = 23$
  - Formula: $q_{\text{in}} = \operatorname{clip}\left(\operatorname{round}(x / S_{\text{in}}) + Z_{\text{in}}, -128, 127\right)$
- **Output Tensor Contract:**
  - Shape: `[1, 40]`, dtype `int8`
  - Scale: $S_{\text{out}} = 0.00390625$ ($1.0 / 256.0$)
  - Zero Point: $Z_{\text{out}} = -128$
  - Dequantization Formula: $p = (q_{\text{out}} - Z_{\text{out}}) \times S_{\text{out}} = (q_{\text{out}} + 128) / 256.0 \in [0.0, 1.0]$
  - Monotonic Property: $\operatorname{argmax}(q_{\text{out}}) \equiv \operatorname{argmax}(p)$.

#### Test Case Predictions & Top-5 Outcomes:
| Case Key | Category | Predicted Sign | Conf | Top-5 Ranking Hierarchy |
| :--- | :--- | :---: | :---: | :--- |
| `case_1_canonical_all_zeros` | Idle Baseline (all bytes = 23) | `orange` (id 30) | 0.3672 | 1: `orange` (0.3672), 2: `thin` (0.1719), 3: `chair` (0.0977), 4: `dog` (0.0898), 5: `cool` (0.0742) |
| `case_2_extreme_bounds` | Saturated `[-128, 127]` | `finish` (id 19) | 0.4727 | 1: `finish` (0.4727), 2: `thanksgiving` (0.4570), 3: `chair` (0.0273), 4: `cool` (0.0156), 5: `candy` (0.0117) |
| `case_3_single_hand_motion` | Synthetic 1-Hand Sign | `black` (id 3) | 0.8047 | 1: `black` (0.8047), 2: `bed` (0.1562), 3: `cool` (0.0195), 4: `blue` (0.0078), 5: `drink` (0.0078) |
| `case_4_two_hand_motion` | Synthetic 2-Hand Sign | `cool` (id 12) | 0.7383 | 1: `cool` (0.7383), 2: `thanksgiving` (0.2500), 3: `finish` (0.0039), 4: `clothes` (0.0039), 5: `chair` (0.0) |
| `case_5_real_test_sample_1` | Held-out Video (GT: `book`) | `book` (id 5) | **0.9805** | 1: `book` (0.9805), 2: `family` (0.0195), 3: `yes` (0.0), 4: `clothes` (0.0), 5: `drink` (0.0) |
| `case_5_real_test_sample_2` | Held-out Video (GT: `clothes`) | `clothes` (id 10) | **0.9961** | 1: `clothes` (0.9961), 2: `yes` (0.0), 3: `year` (0.0), 4: `family` (0.0), 5: `drink` (0.0) |
| `case_5_real_test_sample_3` | Held-out Video (GT: `chair`) | `chair` (id 9) | **0.9961** | 1: `chair` (0.9961), 2: `year` (0.0), 3: `family` (0.0), 4: `drink` (0.0), 5: `dog` (0.0) |

---

## 4. Master Orchestration & Determinism Verification

The Main Agent implemented the master orchestration script [tests/fixtures/android_parity/generate_all_fixtures.py](file:///c:/EchoCam/tests/fixtures/android_parity/generate_all_fixtures.py), which executes the pipeline end-to-end and validates determinism across consecutive runs.

### 4.1. Deterministic Checksum Parity Across Regeneration
| Artifact File | Size | Run 1 SHA-256 Checksum | Run 2 SHA-256 Checksum | Parity Status |
| :--- | :---: | :--- | :--- | :---: |
| `stage1_features.npz` | 5,834 B | `A0AB0DD889A23CE7329E3B26F6F4124605615822182B0FE192CF1BA21BD8FAE1` | `A0AB0DD889A23CE7329E3B26F6F4124605615822182B0FE192CF1BA21BD8FAE1` | **BIT-FOR-BIT IDENTICAL** |
| `stage2_temporal.npz` | 1,091,283 B | `E3B9FB3A6BC1B21EF61C6E757CB56F407EFD694B52D75995D2BADAE4DF1BE4DB` | `E3B9FB3A6BC1B21EF61C6E757CB56F407EFD694B52D75995D2BADAE4DF1BE4DB` | **BIT-FOR-BIT IDENTICAL** |
| `stage3_inference.npz` | 682,841 B | `ACA2611EEB21B962B26876BDEA20FC6E0D4447B75C1BE8F681B2A450348E137B` | `ACA2611EEB21B962B26876BDEA20FC6E0D4447B75C1BE8F681B2A450348E137B` | **BIT-FOR-BIT IDENTICAL** |

---

## 5. Automated Validation Test Suite

A comprehensive automated test suite was constructed in [tests/test_parity_fixtures.py](file:///c:/EchoCam/tests/test_parity_fixtures.py). Execution results via `python -m unittest`:

```
Ran 8 tests in 0.240s
OK
```

Full repository test suite discovery (`python -m unittest discover -s tests -p "test_*.py"`):
```
Ran 75 tests in 7.384s
OK
```

Android JVM unit test suite (`cd android && .\gradlew.bat testDebugUnitTest`):
```
BUILD SUCCESSFUL in 12s
25 actionable tasks: 25 up-to-date (6/6 tests passed)
```

---

## 6. Phase 7B.1 Sign-Off & Strict Phase Boundary

| Audit Category | Evaluation Result | Evidence |
| :--- | :--- | :--- |
| **Pipeline Representation** | **PASS** | Generated directly from production code (`LiveLandmarkExtractor`, `resample_sequence`, `EchoGestureRuntime`). |
| **Dimensional Correctness** | **PASS** | Stage 1 (134-D), Stage 2 (64, 263), Stage 3 ([1, 64, 263] $\to$ [1, 40]). |
| **Numerical Sanity** | **PASS** | Zero NaNs, zero Infs across all 23 test cases. |
| **Zero-Masking Invariants** | **PASS** | Inactive hand slots, missing hands, and idle windows strictly $0.0$. |
| **Transition Spike Prevention** | **PASS** | Hand appearances and disappearances enforce $0.0$ velocity. |
| **Quantization Parity** | **PASS** | Affine scale/zero-point formulas and monotonic argmax verified bit-exact. |
| **Model & Split Integrity** | **PASS** | Model SHA-256 (`ABD726BC...`) and dataset splits unchanged. |

**Strict Execution Boundary:** Phase 7B.1 is complete. No Kotlin recognition pipeline files have been created. No APK installation or CameraX inference binding has occurred. Execution stops here awaiting authorization for Phase 7B.2 (Kotlin Recognition Pipeline Implementation).
