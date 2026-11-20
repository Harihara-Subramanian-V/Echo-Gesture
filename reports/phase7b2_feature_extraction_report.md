# EchoGesture Phase 7B.2: Kotlin 134-D Feature Extraction Port Report

**Project:** `C:\EchoCam`  
**Android Workspace:** `C:\EchoCam\android`  
**Phase:** 7B.2 (Kotlin Stage 1 Feature Extraction Port)  
**Date:** November 20, 2026  
**Status:** COMPLETE & CERTIFIED (100% Bit-Level Parity)

---

## 1. Executive Summary

Phase 7B.2 ports the first stage of the frozen Python sign language recognition pipeline into native Kotlin within the Android application architecture.

The objective was to achieve bit-level numerical parity between the reference Python implementation (`scripts/echogesture_runtime.py`, `scripts/live_inference.py`) and the native Android Kotlin implementation.

Using the deterministic Stage 1 reference fixtures generated in Phase 7B.1 (`tests/fixtures/android_parity/stage1_features.json`), the Kotlin port was verified across all 7 canonical test cases (comprising single-hand, two-hand, missing-hand, temporal hand transition, Hungarian tracking continuity, and varying aspect ratio conditions).

### Key Results
- **Numerical Parity**: **100% exact parity** across all 14 evaluated fixture frames/states.
  - **Maximum Absolute Error**: `0.000000e+00`
  - **Mean Absolute Error**: `0.000000e+00`
  - **Mismatches ($> 10^{-6}$)**: `0 / 134`
- **Kotlin Unit Test Suite**: **37 / 37 tests PASSED** (0 failures, 0 errors, 0 skipped).
  - `HandSlotResolverTest`: 14 / 14 passed
  - `FeatureExtractor134DTest`: 10 / 10 passed
  - `Stage1ParityTest`: 7 / 7 passed
  - `ModelContractTest`: 6 / 6 passed
- **Android Build**: `.\gradlew.bat assembleDebug` **SUCCESSFUL** in 12s.
- **Python Master Test Suite**: **75 / 75 tests PASSED** in 7.67s.
- **Frozen Model SHA-256**: `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` (Unmodified).

---

## 2. Architecture & Code Structure

The Stage 1 feature pipeline converts raw hand candidate detections from MediaPipe into canonical 134-dimensional feature vectors:
$$\text{List<DetectedHandCandidate>} \xrightarrow{\text{HandSlotResolver}} \text{ResolvedHandSlots} \xrightarrow{\text{FeatureExtractor134D}} \text{FloatArray(134)}$$

### 2.1 HandSlotResolver (`com.echogesture.app.feature.HandSlotResolver`)
- **Location**: `android/app/src/main/java/com/echogesture/app/feature/HandSlotResolver.kt`
- **Data Models**:
  - `DetectedHandCandidate(landmarks: FloatArray, handednessLabel: String, score: Float)`
    - Represents a detected hand from MediaPipe with 63 raw normalized coordinates (21 landmarks $\times [x, y, z]$).
  - `ResolvedHandSlots(leftHand: DetectedHandCandidate?, rightHand: DetectedHandCandidate?)`
    - Encapsulates canonical slot assignments where Slot 0 is strictly the **Viewer-Left Hand** and Slot 1 is the **Viewer-Right Hand**.
- **Slot Assignment Logic**:
  - **Zero Candidates**: Both slots set to `null`.
  - **Single Candidate**:
    - If label is `"Left"`: Slot 0 = candidate, Slot 1 = `null`.
    - If label is `"Right"`: Slot 0 = `null`, Slot 1 = candidate.
    - Fallback: Assigned based on wrist $x$ position ($x \le 0.5 \implies$ Slot 0, $x > 0.5 \implies$ Slot 1).
  - **Two Candidates (Distinct Labels)**:
    - Candidate with label `"Left"` assigned to Slot 0; candidate with label `"Right"` assigned to Slot 1.
    - If duplicate labels exist with different scores, highest confidence candidate takes the matching slot.
  - **Two Candidates (Ambiguous / Duplicate Labels)**:
    - **Hungarian Temporal Tracking**: When both hands were present in the immediate previous frame ($t-1$), Euclidean 2D distances between current wrists $(x, y)$ and previous wrists are computed:
      $$d_{\text{direct}} = \|w_0 - w_{\text{prev0}}\|_2 + \|w_1 - w_{\text{prev1}}\|_2$$
      $$d_{\text{swap}} = \|w_0 - w_{\text{prev1}}\|_2 + \|w_1 - w_{\text{prev0}}\|_2$$
      If $d_{\text{swap}} < d_{\text{direct}}$, slots are swapped to maintain smooth temporal trajectory without ID jumping.
    - **Horizontal Fallback**: When temporal tracking is unavailable (e.g. at gesture onset), the candidate with smaller wrist $x$ coordinate is assigned to Viewer-Left (Slot 0) and larger wrist $x$ to Viewer-Right (Slot 1).
  - **Three or More Candidates**: Filtered to the two highest-confidence candidates before running the two-candidate resolution algorithm.
  - **State Management**: Maintains `prevWrists` and `prevPresence` across frames, with an explicit `reset()` method to clear temporal state upon stream interruption or idle reset.

### 2.2 FeatureExtractor134D (`com.echogesture.app.feature.FeatureExtractor134D`)
- **Location**: `android/app/src/main/java/com/echogesture/app/feature/FeatureExtractor134D.kt`
- **134-D Output Vector Specification**:
  ```
  [0   : 63 ] : Slot 0 (Viewer-Left) 21 landmarks x (x, y, z)
  [63  : 126] : Slot 1 (Viewer-Right) 21 landmarks x (x, y, z)
  [126      ] : Presence flag Left (1.0f if present, 0.0f if missing)
  [127      ] : Presence flag Right (1.0f if present, 0.0f if missing)
  [128      ] : Handedness confidence score Left ([0.0f, 1.0f], 0.0f if missing)
  [129      ] : Handedness confidence score Right ([0.0f, 1.0f], 0.0f if missing)
  [130      ] : Co-presence flag (1.0f if BOTH hands present, 0.0f otherwise)
  [131      ] : Geometric inter-wrist distance (normalized by diagonal)
  [132      ] : Horizontal wrist offset (dx_px / diagonal)
  [133      ] : Vertical wrist offset (dy_px / diagonal)
  ```
- **Aspect-Ratio Corrected Diagonal Normalization**:
  To guarantee aspect-ratio invariance across camera sensor resolutions:
  $$\text{diag} = \sqrt{W^2 + H^2}$$
  $$\Delta x_{\text{px}} = (x_{w1} - x_{w0}) \cdot W, \quad \Delta y_{\text{px}} = (y_{w1} - y_{w0}) \cdot H$$
  $$d_{\text{geom}} = \frac{\sqrt{\Delta x_{\text{px}}^2 + \Delta y_{\text{px}}^2}}{\text{diag}}$$
  $$\text{offset}_x = \frac{\Delta x_{\text{px}}}{\text{diag}}, \quad \text{offset}_y = \frac{\Delta y_{\text{px}}}{\text{diag}}$$
- **Strict Inactive Zero-Masking**:
  - Inactive hand landmark coordinates are strictly `0.0f`.
  - Inactive handedness scores and presence flags are strictly `0.0f`.
  - When either hand is missing ($P_L = 0$ or $P_R = 0$):
    $$\text{vector}[130..133] \equiv 0.0\text{f}$$
    (Co-presence flag, geometric distance, and wrist offsets are strictly forced to 0.0).
- **High-Performance Memory Optimization**:
  - Supports zero-allocation in-place buffer filling via `extractTo(slots, width, height, destination: FloatArray)` to prevent garbage collection pauses during real-time 30 FPS camera processing.

---

## 3. Parity Validation Evidence

Automated validation was performed by `Stage1ParityTest.kt` directly reading `tests/fixtures/android_parity/stage1_features.json`.

### 3.1 Results per Canonical Test Case

| Case ID | Description | Frames | Inactive Zero-Masking | Flags Parity | Max Abs Error | Mean Abs Error | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `case_1_left_only` | Isolated Left Hand | 1 | Slot 1 [63:126] $\equiv 0.0$ | $P_L=1, P_R=0, P_{\text{both}}=0$ | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_2_right_only` | Isolated Right Hand | 1 | Slot 0 [0:63] $\equiv 0.0$ | $P_L=0, P_R=1, P_{\text{both}}=0$ | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_3_both_hands` | Two-Handed Signing | 1 | Active | $P_L=1, P_R=1, P_{\text{both}}=1$ | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_4_missing_hands` | No Hands Detected | 1 | All 134 values $\equiv 0.0$ | $P_L=0, P_R=0, P_{\text{both}}=0$ | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_5_transition_sequence` | 5-Frame Transition Sequence | 5 | Masked when inactive | Verified per frame | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_6_hungarian_tracking` | Swap Distance Resolution | 2 | Continuity maintained | Verified per frame | `0.000000e+00` | `0.000000e+00` | **PASS** |
| `case_7_aspect_ratios` | 4:3 vs 16:9 Invariance | 3 | Resolution scaled | Verified per resolution | `0.000000e+00` | `0.000000e+00` | **PASS** |

### 3.2 Key Parity Insights
1. **Aspect Ratio Invariance**:
   For $1280 \times 720$ and $1920 \times 1080$, the normalized inter-wrist distance ($0.3486302$) and offsets are identical down to the lowest bit because both resolutions share the exact 16:9 aspect ratio. For $640 \times 480$ (4:3), the diagonal scaling formula correctly maps the pixel offsets.
2. **Hungarian Temporal Consistency**:
   Under ambiguous duplicate label assignments, the Hungarian swap check ($d_{\text{swap}} < d_{\text{direct}}$) prevents trajectory jumping, exactly matching the Python tracker behavior.

---

## 4. Test Suite Execution Logs

### 4.1 Android Unit Tests (Gradle)
```text
PS C:\EchoCam\android> .\gradlew.bat testDebugUnitTest

BUILD SUCCESSFUL in 10s
25 actionable tasks: 25 up-to-date
```

**JUnit XML Reports:**
- `TEST-com.echogesture.app.feature.Stage1ParityTest.xml`: 7 tests, 0 failures, 0 errors, 0 skipped
- `TEST-com.echogesture.app.feature.HandSlotResolverTest.xml`: 14 tests, 0 failures, 0 errors, 0 skipped
- `TEST-com.echogesture.app.feature.FeatureExtractor134DTest.xml`: 10 tests, 0 failures, 0 errors, 0 skipped
- `TEST-com.echogesture.app.ModelContractTest.xml`: 6 tests, 0 failures, 0 errors, 0 skipped
- **Total Kotlin Unit Tests**: 37 / 37 passed.

### 4.2 Android Build Assembly (Gradle)
```text
PS C:\EchoCam\android> .\gradlew.bat assembleDebug

BUILD SUCCESSFUL in 12s
38 actionable tasks: 4 executed, 34 up-to-date
```

### 4.3 Master Python Test Suite
```text
PS C:\EchoCam> & "C:\EchoCam\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
----------------------------------------------------------------------
Ran 75 tests in 7.673s

OK
```

### 4.4 Frozen Model Integrity
```text
PS C:\EchoCam> Get-FileHash models/quantized/echogesture_exp_d_int8.tflite -Algorithm SHA256

Algorithm: SHA256
Hash:      ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499
Path:      C:\EchoCam\models\quantized\echogesture_exp_d_int8.tflite
```

---

## 5. Phase Sign-Off & Next Steps

Phase 7B.2 is complete and certified. The Stage 1 (134-D feature extraction) Kotlin port is identical to the Python pipeline.

### Next Planned Phase: Phase 7B.3 (Temporal Preprocessing Port)
Phase 7B.3 will port Stage 2 of the pipeline:
1. Dynamic sequence resampling (incomplete-window padding and uniform linear interpolation to $T=32$ frames).
2. Standard scaler feature normalization ($z = (x - \mu) / \sigma$) using frozen parameters from `data/processed/sequences_v3/scaler_params.npz`.
3. First-order velocity delta calculation ($\Delta f_t = f_t - f_{t-1}$, with boundary $\Delta f_0 = 0$).
4. 263-D temporal tensor assembly matching `tests/fixtures/android_parity/stage2_temporal.json`.
