# EchoGesture Phase 7B.3: Kotlin Temporal Preprocessing Port (Stage 2) Report

**Project:** `C:\EchoCam`  
**Android Workspace:** `C:\EchoCam\android`  
**Phase:** 7B.3 (Kotlin Stage 2 Temporal Preprocessing Port)  
**Date:** November 20, 2026  
**Status:** COMPLETE & CERTIFIED (100% Numerical Parity)

---

## 1. Executive Summary

Phase 7B.3 ports Stage 2 of the frozen Python sign language recognition pipeline into native Kotlin within the Android application architecture (`com.echogesture.app.preprocessing`).

Stage 2 executes the complete temporal preprocessing transformation:
$$\text{Raw Features } [T, 134] \xrightarrow{\text{TemporalResampler}} [64, 134] \xrightarrow{\text{SequenceNormalizer}} [64, 134] \xrightarrow{\text{VelocityFeatureExtractor}} [64, 129] \xrightarrow{\text{Concatenation}} [64, 263]$$

The implementation was validated against the authoritative Stage 2 Python reference fixtures (`tests/fixtures/android_parity/stage2_temporal.json` and `.npz`) covering all 8 canonical test cases (short sequences, full sequences, downsampled long sequences, sparse detections, single-hand continuous, two-hand continuous, alternating hands transition, and idle windows).

### Key Verification Metrics
- **Evaluated Fixture Cases**: 8 cases ($472,576$ individual floating-point elements evaluated).
- **Mismatches Exceeding Tolerance ($> 2.0 \times 10^{-5}$)**: **`0 / 472,576` (100% numerical agreement)**.
- **Stage Breakdown Precision**:
  - `TemporalResampler` (134-D): Max Absolute Error = $1.192093 \times 10^{-7}$ ($\le 1$ ULP).
  - `SequenceNormalizer` (134-D): Max Absolute Error = $7.241964 \times 10^{-6}$.
  - `VelocityFeatureExtractor` (129-D): Max Absolute Error = $1.583248 \times 10^{-7}$.
  - `263-D Output Concatenation`: Max Absolute Error = $7.241964 \times 10^{-6}$.
  - `End-to-End Pipeline Chain`: Global Max Absolute Error = $1.264364 \times 10^{-5}$.
- **Android Unit Tests**: **89 / 89 tests PASSED** (0 failures, 0 errors, 0 skipped).
  - `Stage2ParityTest`: 9 / 9 passed
  - `TemporalResamplerTest`: 16 / 16 passed
  - `SequenceNormalizerTest`: 12 / 12 passed
  - `VelocityFeatureExtractorTest`: 11 / 11 passed
  - `SequencePreprocessor263DTest`: 4 / 4 passed
  - Stage 1 & Architecture tests: 37 / 37 passed
- **Android Build**: `.\gradlew.bat assembleDebug` **SUCCESSFUL** in 12s.
- **Python Master Test Suite**: **75 / 75 tests PASSED** in 7.11s.
- **Frozen Model SHA-256**: `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` (Unmodified).

---

## 2. Architecture & Code Structure

The Stage 2 preprocessing subsystem resides under `com.echogesture.app.preprocessing`:

```
android/app/src/main/java/com/echogesture/app/preprocessing/
├── TemporalResampler.kt          # Variable T -> fixed 64 frames resampling
├── SequenceNormalizer.kt         # Active-frame Z-score standardization
├── VelocityFeatureExtractor.kt   # 129-D temporal velocity with spike prevention
└── SequencePreprocessor263D.kt   # Fused end-to-end preprocessing entry point
```

### 2.1 TemporalResampler (`TemporalResampler.kt`)
- **Ported from**: `scripts/prepare_sequences.py:resample_sequence`, `resample_hand_landmarks`, `resample_interaction_features`.
- **Target Time Grid**:
  - $T \le 1$: $target\_t = [0.0, \dots, 0.0]$.
  - $T > 1$: $target\_t[j] = j \cdot \frac{T - 1}{63.0\text{f}}$ for $j \in [0, 63]$.
- **Nearest-Neighbor Presence Sampling**:
  - $nearest\_idx[j] = \text{round}(target\_t[j])\text{.toInt().coerceIn}(0, T - 1)$.
  - Uses `kotlin.math.round` (IEEE-754 round-to-nearest-even / `rint`, matching `np.round`).
  - $target\_pres[j] = 1.0\text{f}$ if $sequence[nearest\_idx[j]][pres\_idx] > 0.5\text{f}$, else $0.0\text{f}$.
- **Absence-Aware Trajectory Interpolation**:
  - 0 active frames: coordinates and score remain strictly $0.0\text{f}$.
  - 1 active frame: propagated to target frames where $target\_pres[j] > 0.5\text{f}$.
  - $\ge 2$ active frames: linear interpolation in `Double` precision along active frames:
    $$\frac{dy}{dx} = \frac{f_{i+1} - f_i}{x_{i+1} - x_i}, \quad y = f_i + (x - x_i) \cdot \frac{dy}{dx}$$
    Output value converted to `Float` and strictly zero-clamped where $target\_pres[j] \le 0.5\text{f}$.
- **Interaction Features Resampling**:
  - $target\_both[j] = 1.0\text{f}$ if $(target\_pres\_L[j] > 0.5\text{f} \land target\_pres\_R[j] > 0.5\text{f})$, else $0.0\text{f}$.
  - Interpolated strictly along co-present frames; clamped to $0.0\text{f}$ where $target\_both[j] \le 0.5\text{f}$.

### 2.2 SequenceNormalizer (`SequenceNormalizer.kt`)
- **Ported from**: `scripts/prepare_sequences.py:apply_scaler` and `scripts/live_inference.py:LiveTemporalPreprocessor.process_window`.
- **Frozen Parameters**: Loaded from `android/app/src/main/assets/scaler_params.json` (exported and verified from `data/processed/sequences_v3/scaler_params.npz`).
- **Standardization Formula**:
  - Active Left Hand ($P_L > 0.5$): $x'_c = \frac{x_c - \mu_c}{\sigma_c + 10^{-6}\text{f}}$ for $c \in [0, 62]$; absent $\implies 0.0\text{f}$.
  - Active Right Hand ($P_R > 0.5$): $x'_c = \frac{x_c - \mu_c}{\sigma_c + 10^{-6}\text{f}}$ for $c \in [63, 125]$; absent $\implies 0.0\text{f}$.
  - Unscaled Metadata ($c \in [126, 130]$): $P_L, P_R, S_L, S_R, P_{\text{both}}$ pass through unmodified.
  - Active Interaction ($P_{\text{both}} > 0.5$): $x'_c = \frac{x_c - \mu_c}{\sigma_c + 10^{-6}\text{f}}$ for $c \in [131, 133]$; absent $\implies 0.0\text{f}$.
- **Zero-Allocation**: Accepts preallocated `destination` buffer and supports safe in-place normalization.

### 2.3 VelocityFeatureExtractor (`VelocityFeatureExtractor.kt`)
- **Ported from**: `scripts/augment_sequences.py:compute_velocity_features` and `scripts/live_inference.py:LiveTemporalPreprocessor.process_window`.
- **Channel Layout (129-D)**:
  - `[0..62]`: Left hand landmark velocities (63 channels).
  - `[63..125]`: Right hand landmark velocities (63 channels).
  - `[126..128]`: Two-hand interaction velocities (3 channels: wrist distance, $dx$, $dy$).
- **Invariants**:
  - **Frame 0 Boundary**: Strictly $0.0\text{f}$ across all 129 velocity channels for $t = 0$.
  - **Transition Spike Prevention**: For $t \in [1, 64)$, velocity is computed if and only if both consecutive frames $t$ and $t-1$ are active ($> 0.5\text{f}$). On hand appearance or disappearance, velocity is strictly clamped to $0.0\text{f}$.

### 2.4 SequencePreprocessor263D (`SequencePreprocessor263D.kt`)
- **Fused Pipeline**: Coordinates `TemporalResampler`, `SequenceNormalizer`, and `VelocityFeatureExtractor` in sequential order.
- **Output Formats**:
  - 2D Tensor: `Array<FloatArray>` of shape `[64, 263]`.
  - Flattened Buffer: `FloatArray` of length $64 \times 263 = 16,832$ for direct zero-copy LiteRT buffer population.
- **Memory Management**: Pre-allocates all intermediate buffers (`resampledBuffer`, `normalizedBuffer`, `velocityBuffer`, `outputBuffer`) on instantiation, eliminating GC pauses during real-time 30 FPS camera inference.

---

## 3. Scaler Parameters Export & Verification

A dedicated reproducible export tool was authored: `scripts/export_scaler_params.py`.

```
Source: data/processed/sequences_v3/scaler_params.npz (Compressed NumPy format)
Target: android/app/src/main/assets/scaler_params.json (Human-auditable JSON)
Size:   7,002 bytes
```

### Verification Evidence:
- Mean vector: 134 floats, verified exact float32 equality against source NPZ.
- Standard deviation vector: 134 floats, verified exact float32 equality against source NPZ.
- Channels 126..130: Mean is strictly $0.0$, Std is strictly $1.0$.
- No NaN, Inf, or non-positive standard deviation values detected.

---

## 4. Stage 2 Parity Validation Results

Validation was conducted using `Stage2ParityTest.kt` loading the Python reference fixtures (`tests/fixtures/android_parity/stage2_temporal.json`).

### 4.1 Quantitative Parity Results Matrix across all 8 Cases

| Case ID | Name | $T_{in}$ | Resampler Max Err | Normalizer Max Err | Velocity Max Err | Concat 263 Max Err | Full Chain Max Err | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `case_1_short_sequence` | 24 | $1.192 \times 10^{-7}$ | $7.242 \times 10^{-6}$ | $1.565 \times 10^{-7}$ | $7.242 \times 10^{-6}$ | $6.742 \times 10^{-6}$ | **PASS** |
| **2** | `case_2_full_sequence` | 64 | $0.000$ | $7.153 \times 10^{-6}$ | $1.574 \times 10^{-7}$ | $7.153 \times 10^{-6}$ | $1.264 \times 10^{-5}$ | **PASS** |
| **3** | `case_3_long_sequence` | 96 | $1.192 \times 10^{-7}$ | $6.855 \times 10^{-6}$ | $1.583 \times 10^{-7}$ | $6.855 \times 10^{-6}$ | $8.956 \times 10^{-6}$ | **PASS** |
| **4** | `case_4_sparse_detections` | 50 | $1.192 \times 10^{-7}$ | $7.153 \times 10^{-6}$ | $1.490 \times 10^{-7}$ | $7.153 \times 10^{-6}$ | $5.811 \times 10^{-6}$ | **PASS** |
| **5** | `case_5_single_hand_continuous` | 64 | $0.000$ | $6.676 \times 10^{-6}$ | $1.583 \times 10^{-7}$ | $6.676 \times 10^{-6}$ | $7.845 \times 10^{-6}$ | **PASS** |
| **6** | `case_6_two_hand_continuous` | 64 | $0.000$ | $7.033 \times 10^{-6}$ | $1.579 \times 10^{-7}$ | $7.033 \times 10^{-6}$ | $1.213 \times 10^{-5}$ | **PASS** |
| **7** | `case_7_alternating_hands` | 64 | $0.000$ | $6.706 \times 10^{-6}$ | $1.565 \times 10^{-7}$ | $6.706 \times 10^{-6}$ | $1.188 \times 10^{-5}$ | **PASS** |
| **8** | `case_8_idle_window` | 64 | $0.000$ | $0.000$ | $0.000$ | $0.000$ | $0.000$ | **PASS** |

### 4.2 Invariant Verification
1. **Tensor Shape**: All outputs strictly have shape `[64, 263]` (16,832 floats).
2. **Binary Presence**: Channels `126, 127, 130` are strictly `0.0f` or `1.0f`.
3. **Strict Zero-Clamping**: Inactive hands and inactive interaction channels are strictly `0.0f`.
4. **Frame 0 Velocity**: Strictly `0.0f` across all 129 channels for all cases.
5. **Transition Spike Prevention**: At frame 32 in `case_7_alternating_hands`, disappearance and appearance velocities are strictly `0.0f` (no spike).
6. **Idle Sequence**: `case_8_idle_window` outputs strictly all zeros ($0.0\text{f}$) across all 16,832 elements.

---

## 5. Test Suite Execution Logs

### 5.1 Android Unit Tests (Gradle)
```text
PS C:\EchoCam\android> .\gradlew.bat testDebugUnitTest

BUILD SUCCESSFUL in 24s
25 actionable tasks: 2 executed, 23 up-to-date
```

**JUnit XML Reports Summary:**
| Suite Class | Tests | Failures | Errors | Skipped | Time |
|---|:---:|:---:|:---:|:---:|:---:|
| `feature.FeatureExtractor134DTest` | 10 | 0 | 0 | 0 | 0.011s |
| `feature.HandSlotResolverTest` | 14 | 0 | 0 | 0 | 0.016s |
| `feature.Stage1ParityTest` | 7 | 0 | 0 | 0 | 0.065s |
| `ModelContractTest` | 6 | 0 | 0 | 0 | 0.041s |
| `preprocessing.TemporalResamplerTest` | 16 | 0 | 0 | 0 | 1.829s |
| `preprocessing.SequenceNormalizerTest` | 12 | 0 | 0 | 0 | 3.614s |
| `preprocessing.VelocityFeatureExtractorTest` | 11 | 0 | 0 | 0 | 1.243s |
| `preprocessing.Stage2ParityTest` | 9 | 0 | 0 | 0 | 0.448s |
| `preprocessing.SequencePreprocessor263DTest` | 4 | 0 | 0 | 0 | 0.849s |
| **TOTAL** | **89** | **0** | **0** | **0** | **8.116s** |

### 5.2 Android Build Assembly (Gradle)
```text
PS C:\EchoCam\android> .\gradlew.bat assembleDebug

BUILD SUCCESSFUL in 12s
38 actionable tasks: 6 executed, 32 up-to-date
```

### 5.3 Python Master Test Suite
```text
PS C:\EchoCam> & "C:\EchoCam\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"
----------------------------------------------------------------------
Ran 75 tests in 7.114s

OK
```

### 5.4 Frozen Model Integrity
```text
PS C:\EchoCam> Get-FileHash models/quantized/echogesture_exp_d_int8.tflite -Algorithm SHA256

Algorithm: SHA256
Hash:      ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499
Path:      C:\EchoCam\models\quantized\echogesture_exp_d_int8.tflite
```

---

## 6. Known Limitations & Notes

1. **Floating-Point Tolerances**:
   - The end-to-end chain exhibits maximum error $\approx 1.26 \times 10^{-5}$ compared to Python reference output. This tiny difference stems from float32 division and subtraction ordering between NumPy and the JVM, but is orders of magnitude smaller than the INT8 input quantization step ($S_{\text{in}} = 0.05794$), meaning it will quantize to identical INT8 values.
2. **Preprocessing Only**:
   - This phase covers temporal sequence preprocessing only. It does not perform on-device LiteRT inference (Phase 7B.4) or live camera feed integration (Phase 7C).

---

## 7. Phase Sign-Off & Next Steps

Phase 7B.3 is complete and certified. Stage 1 (134-D feature extraction) and Stage 2 (temporal preprocessing) are fully functional in native Kotlin.

### Next Planned Phase: Phase 7B.4 (LiteRT INT8 On-Device Inference Integration)
Phase 7B.4 will:
1. Load `echogesture_exp_d_int8.tflite` via LiteRT Interpreter.
2. Implement affine input quantization ($q = \text{clip}(\text{round}(X / S_{\text{in}}) + Z_{\text{in}}, -128, 127)$).
3. Execute the INT8 graph to obtain 40-class output logits.
4. Implement affine dequantization and Top-5 class ranking matching `tests/fixtures/android_parity/stage3_inference.json`.
