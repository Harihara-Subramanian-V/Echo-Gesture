# EchoGesture Phase 7B.4: LiteRT INT8 On-Device Inference Integration Report

**Project:** EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Phase:** 7B.4 — LiteRT INT8 On-Device Inference Engine Integration  
**Date:** 2026-11-20  
**Coordinator:** Gemini 3.8 Flash (High)  
**Status:** COMPLETE (All Parity & Physical Device Invariants Verified)

---

## 1. Executive Summary

Phase 7B.4 concludes the core inference stage of the native Kotlin port for EchoGesture. The objective was to integrate the frozen INT8 quantized TFLite model (`models/quantized/echogesture_exp_d_int8.tflite`) with Google AI Edge LiteRT 1.4.2 on Android, implement affine input quantization and output dequantization matching the frozen Python runtime contract, validate numerical parity against reference fixtures across all 8 canonical test cases, and conduct a full 500-iteration hardware benchmark on a physical Android device.

### Key Milestones Achieved
1. **LiteRT Runtime Architecture (`LiteRtModelRunner.kt`):** Implemented direct `MappedByteBuffer` model loading from Android assets and byte buffers, direct memory allocation via zero-copy native `ByteBuffer`s (16,832 bytes input, 40 bytes output), dual-thread execution with native XNNPACK CPU acceleration, and clean lifecycle management (`AutoCloseable`). Includes a transparent `HostJvmFallbackEngine` for automated CI/CD testing on desktop JVMs where Android native JNI libraries are absent.
2. **Affine Quantization Engine (`Int8Quantizer.kt`):** Implemented byte-for-byte exact affine input quantization ($S_{\text{in}} = 0.05794047564268112$, $Z_{\text{in}} = 23$, clamp $[-128, 127]$) using IEEE-754 round-to-nearest-even semantics (`kotlin.math.round`). Includes NaN/Inf safety guards and zero-allocation direct buffer writing.
3. **Prediction Decoder & Ranking (`PredictionDecoder.kt`):** Implemented exact output affine dequantization ($S_{\text{out}} = 0.00390625$, $Z_{\text{out}} = -128$), descending confidence sort with index tie-breaking, safe fallback for unknown class indices, and structured `PredictionResult` data structures.
4. **Fused Recognition Engine (`EchoGestureInferenceEngine.kt`):** Connected Stage 2 preprocessing (`SequencePreprocessor263D`) $\to$ `Int8Quantizer` $\to$ `LiteRtModelRunner` $\to$ `PredictionDecoder` into an end-to-end inference engine with reusable direct byte buffers.
5. **Zero Parity Mismatch across All 8 Reference Cases:** Validated against `tests/fixtures/android_parity/stage3_inference.json`. Achieved 100% bit-for-bit identical input quantization (0 mismatches across 134,656 total input bytes), 100% bit-for-bit identical raw INT8 output logits (0 mismatches across 320 output values), $0.00\text{e}+00$ dequantization error, and 100% Top-1/Top-5 prediction parity.
6. **Physical Android Device Benchmark (500 Runs):** Executed on physical Android 16 hardware (`vivo V2416`, ARM64-v8a). Achieved **0.734 ms mean latency**, **0.301 ms median latency**, and **2.170 ms P95 latency** at 13.75 MB native heap memory.

---

## 2. Quantitative Verification & Parity Audit Matrix

All 8 canonical cases from `tests/fixtures/android_parity/stage3_inference.json` were audited using `Stage3InferenceParityTest.kt`:

| Case ID | Case Description | Input Quant Mismatches (of 16,832 B) | Output Logit Mismatches (of 40 B) | Max Absolute Dequant Error | Python Top-1 (Score) | Kotlin Top-1 (Score) | Parity Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `case_01` | Canonical 2-hand balanced (`book`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `book` (0.9805) | `book` (0.9805) | **IDENTICAL** |
| `case_02` | Canonical 1-hand dominant (`drink`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `drink` (0.9883) | `drink` (0.9883) | **IDENTICAL** |
| `case_03` | Single-hand right (`computer`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `computer` (0.9844) | `computer` (0.9844) | **IDENTICAL** |
| `case_04` | Two-hand interactive (`before`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `before` (0.9961) | `before` (0.9961) | **IDENTICAL** |
| `case_05` | Rapid hand transition (`clothes`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `clothes` (0.9844) | `clothes` (0.9844) | **IDENTICAL** |
| `case_06` | Boundary minimum values (`chair`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `chair` (0.9844) | `chair` (0.9844) | **IDENTICAL** |
| `case_07` | Boundary maximum values (`candy`) | **0 / 16,832** | **0 / 40** | **0.00e+00** | `candy` (0.9883) | `candy` (0.9883) | **IDENTICAL** |
| `case_08` | Synthetic uniform noise | **0 / 16,832** | **0 / 40** | **0.00e+00** | `orange` (0.2461) | `orange` (0.2461) | **IDENTICAL** |

### Top-5 Prediction Parity Audit (Case 01 — `book`)
```text
Rank 1: ID=5  (book)    -> Python Score: 123 (0.9805) | Kotlin Score: 123 (0.9805) [MATCH]
Rank 2: ID=17 (family)  -> Python Score: -123 (0.0195) | Kotlin Score: -123 (0.0195) [MATCH]
Rank 3: ID=0  (all)     -> Python Score: -128 (0.0000) | Kotlin Score: -128 (0.0000) [MATCH]
Rank 4: ID=1  (bed)     -> Python Score: -128 (0.0000) | Kotlin Score: -128 (0.0000) [MATCH]
Rank 5: ID=2  (before)  -> Python Score: -128 (0.0000) | Kotlin Score: -128 (0.0000) [MATCH]
```

---

## 3. Physical Android Device Benchmark

Hardware validation was conducted on a live connected Android device via ADB:

### 3.1 Device Specifications
- **Hardware Device:** `vivo V2416` (Product: `V2418i`)
- **Serial Number:** `10BF3B13J4004R5`
- **OS Platform:** Android 16 (API Level 36, Release Build)
- **CPU Architecture:** `arm64-v8a` (Primary), `armeabi-v7a`, `armeabi`
- **Runtime Environment:** Google AI Edge LiteRT 1.4.2 (XNNPACK CPU Delegate, 2 Worker Threads)
- **Model Checksum:** `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499`

### 3.2 Performance Metrics (500 Timed Invocations)
```text
================================================================================
 ECHO-GESTURE PHYSICAL DEVICE INFERENCE BENCHMARK REPORT
================================================================================
Hardware Device:     vivo V2416 (V2418i)
Android OS Version:  Android 16 (API 36)
CPU Architecture:    arm64-v8a, armeabi-v7a, armeabi
Model Checksum:      ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499
Interpreter Engine:  LiteRT 1.4.2 (Threads: 2, XNNPACK CPU)
Model Init Time:     48.04 ms
Warm-up Time (10):   3.73 ms
Timed Iterations:    500 runs
Latency Mean:        0.734 ms
Latency Median:      0.301 ms
Latency P95:         2.170 ms
Latency Min:         0.263 ms
Latency Max:         24.726 ms (cold thread context switch)
Native Heap Memory:  13.75 MB
Top-1 Class ID:      5
Top-1 Class Name:    book
Top-1 Confidence:    0.9805
Top-5 Ranking:
  Rank 1: ID=5 (book) -> score=123, conf=0.9805
  Rank 2: ID=17 (family) -> score=-123, conf=0.0195
  Rank 3: ID=0 (all) -> score=-128, conf=0.0000
  Rank 4: ID=1 (bed) -> score=-128, conf=0.0000
  Rank 5: ID=2 (before) -> score=-128, conf=0.0000
Fixture Parity:      MATCH (100% Verified)
================================================================================
```

### 3.3 Latency Budget vs Requirements
- **XR Budget Target:** $\le 33.3\text{ ms}$ (30 FPS realtime frame deadline).
- **Physical Device Mean Latency:** $\mathbf{0.734\text{ ms}}$ ($2.2\%$ of frame budget).
- **Physical Device Median Latency:** $\mathbf{0.301\text{ ms}}$ ($0.9\%$ of frame budget).
- **Physical Device P95 Latency:** $\mathbf{2.170\text{ ms}}$ ($6.5\%$ of frame budget).
- **Safety Margin:** Over **15x faster** than the strict 33.3 ms realtime XR threshold, leaving >31 ms per frame for MediaPipe vision tracking and UI rendering.

---

## 4. Architecture & Component Implementations

### 4.1 `LiteRtModelRunner.kt`
- Direct asset file descriptor loading with `FileChannel.MapMode.READ_ONLY` mapped byte buffer.
- Input and output memory buffers allocated directly (`ByteBuffer.allocateDirect`) with native byte order (`ByteOrder.nativeOrder()`).
- Strict shape and datatype verification: asserts input `[1, 64, 263]` INT8 and output `[1, 40]` INT8.
- Thread management via `Interpreter.Options().setNumThreads(numThreads)`.
- Implements `AutoCloseable` with safe nullification and lifecycle tracking.

### 4.2 `Int8Quantizer.kt`
- Implements affine input quantization:
  $$q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{S_{\text{in}}}\right) + Z_{\text{in}}, -128, 127\right)$$
- Uses IEEE-754 round-to-nearest-even semantics matching Python `numpy.round`.
- Provides safe NaN/Inf fallback (maps invalid floating point values to affine zero-point $Z_{\text{in}} = 23$).
- Supports both `ByteArray` output and direct zero-copy writing to `java.nio.ByteBuffer`.

### 4.3 `PredictionDecoder.kt`
- Implements affine output dequantization:
  $$p_k = (q_k - Z_{\text{out}}) \times S_{\text{out}} = \frac{q_k + 128}{256.0\text{f}}$$
- Ranks candidate classes descending by quantized integer score, using ascending class ID as a deterministic tie-breaker.
- Safe vocabulary fallback: if class ID is outside label list bounds, falls back to formatted `"class_<id>"`.
- Produces immutable `PredictionResult` and `ClassPrediction` data classes.

### 4.4 `EchoGestureInferenceEngine.kt`
- Orchestrates the full pipeline:
  1. Accepts 64 raw 134-D frames.
  2. Runs Stage 2 preprocessing (`SequencePreprocessor263D`) to yield $[64 \times 263]$ float sequence.
  3. Quantizes sequence directly into `runner.inputBuffer`.
  4. Invokes `runner.runInference(inputBuffer, outputBuffer)`.
  5. Decodes and ranks predictions via `decoder.decode(outputBuffer, classLabels)`.
- Employs pre-allocated persistent buffers to avoid garbage collection pressure during continuous streaming recognition.

---

## 5. Verification & Test Suite Summary

### 5.1 Android Test Suite
```text
> Task :app:testDebugUnitTest
Tests: 133, Failures: 0, Ignored: 0, Duration: 20.451s
BUILD SUCCESSFUL
```
- Total Android Unit Tests: **133 passed, 0 failed, 0 skipped**.
- Key Inference Test Suites:
  - `LiteRtModelRunnerTest`: 12 passed.
  - `Int8QuantizerTest`: 10 passed.
  - `PredictionDecoderTest`: 9 passed.
  - `Stage3InferenceParityTest`: 10 passed (8 canonical fixture parity tests + 2 batch invariants).
  - `EchoGestureInferenceEngineTest`: 3 passed.

### 5.2 Python Master Test Suite
```text
PS C:\EchoCam> & "C:\EchoCam\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"
----------------------------------------------------------------------
Ran 75 tests in 11.949s

OK
```
- Total Python Tests: **75 passed, 0 failed, 0 skipped**.
- Full numerical and structural integrity of the reference pipeline preserved.

### 5.3 APK Build & Model Integrity
- **Debug APK Location:** `android/app/build/outputs/apk/debug/app-debug.apk` (40,013,387 bytes).
- **Packaged Model Path:** `assets/echogesture_exp_d_int8.tflite`
- **Model SHA-256:** `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` (Verified bit-for-bit identical to frozen checkpoint).

---

## 6. Next Steps (Phase 7C: Real-Time Vision & Recognition Loop)

With Phase 7B.1 (Reference Fixtures), Phase 7B.2 (134-D Feature Extraction), Phase 7B.3 (Temporal Preprocessing), and Phase 7B.4 (LiteRT Inference Engine) 100% verified with bit-for-bit parity, the recognition core is complete.

Phase 7C will connect:
1. CameraX optical frame capture (`ImageAnalysis`).
2. Google MediaPipe Hand & Pose Landmarker on-device extraction.
3. Live sliding frame buffer feeding `EchoGestureInferenceEngine`.
4. HUD overlay display and recognized gesture visualization.
