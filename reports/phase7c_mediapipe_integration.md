# EchoGesture Phase 7C: MediaPipe Tasks Vision Landmark Integration Report

**Project:** EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Phase:** 7C — Real-Time Vision & Recognition Loop  
**Agent:** Agent 2 — MediaPipe Landmark Specialist  
**Date:** 2026-11-20  
**Status:** COMPLETE & VERIFIED (12/12 Unit Tests Passing)  

---

## 1. Executive Summary

This report documents the architectural design, implementation, and verification of the on-device hand landmark detection subsystem for EchoGesture Phase 7C. The vision extraction layer wraps Google MediaPipe Tasks Vision (`com.google.mediapipe:tasks-vision:0.10.20`), loading the frozen model bundle `hand_landmarker.task` directly from application assets.

The subsystem transforms raw camera frames into normalized, structured `DetectedHandCandidate` representations that feed directly into downstream Hungarian slot matching (`HandSlotResolver`), spatial-temporal feature assembly (`FeatureExtractor134D`), rolling buffer tracking (`LiveTemporalBuffer`), and INT8 quantized sign inference (`LiteRtModelRunner`).

---

## 2. MediaPipe Model & Architecture Configuration

### 2.1 Model Specifications
- **Asset Path:** `android/app/src/main/assets/hand_landmarker.task`
- **File Size:** `7,819,105 bytes` (~7.46 MB)
- **SHA-256 Checksum:** `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`
- **APK Compression:** Configured with `noCompress += "task"` in `android/app/build.gradle.kts` to allow direct memory-mapped asset loading without decompression latency.

### 2.2 Task Configuration Parameters
The detector initializes `com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker` using `HandLandmarkerOptions` configured to match EchoGesture real-time invariants:
- **Running Mode:** `RunningMode.IMAGE`  
  Enables synchronous, deterministic frame-by-frame processing matching CameraX's `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` pipeline.
- **Max Number of Hands:** `setNumHands(2)`  
  Enforces detection of up to 2 simultaneous hands (subject's left and right hands).
- **Minimum Detection Confidence:** `setMinHandDetectionConfidence(0.5f)`  
  Suppresses false positive hand bounding box detections from background clutter.
- **Minimum Presence Confidence:** `setMinHandPresenceConfidence(0.5f)`  
  Ensures hand presence is verified before keypoint regression begins.
- **Minimum Tracking Confidence:** `setMinTrackingConfidence(0.5f)`  
  Maintains landmark tracking consistency across consecutive frames.

### 2.3 Dual-Engine Architecture (Device vs Host JVM)
To support development and continuous integration testing on desktop JVMs (where native Android `libmediapipe_tasks_vision_jni.so` is absent), the system introduces an internal engine abstraction:
- **`IHandLandmarkDetector`**: Public contract defining frame detection and lifecycle operations.
- **`NativeHandLandmarkerEngine`**: Production engine instantiating MediaPipe's native `HandLandmarker` on physical Android runtime.
- **`HostJvmFallbackEngine`**: Safe fallback engine catching `LinkageError` and `UnsatisfiedLinkError` on desktop JVMs, allowing complete unit testing and mock injection without external test stubs.

---

## 3. Coordinate Normalization, Rotation & Ingestion

### 3.1 Landmark Topology & Dimensionality
For each detected hand, MediaPipe outputs 21 standard keypoints:
- **Index 0:** Wrist (Root / Anchor joint)
- **Indices 1–4:** Thumb (CMC, MCP, IP, Tip)
- **Indices 5–8:** Index Finger (MCP, PIP, DIP, Tip)
- **Indices 9–12:** Middle Finger (MCP, PIP, DIP, Tip)
- **Indices 13–16:** Ring Finger (MCP, PIP, DIP, Tip)
- **Indices 17–20:** Pinky Finger (MCP, PIP, DIP, Tip)

Each landmark contains three continuous coordinates:
- $x \in [0.0, 1.0]$: Normalized horizontal coordinate relative to image width.
- $y \in [0.0, 1.0]$: Normalized vertical coordinate relative to image height.
- $z$: Relative depth coordinate relative to the wrist origin ($z=0$ at wrist anchor).

### 3.2 Tensor Flattening Order
The 21 3D coordinates are flattened sequentially into a `FloatArray(63)`:
$$\text{landmarks63} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$
- Offset calculation for landmark index $i \in [0, 20]$:
  - $x_i = \text{landmarks63}[i \times 3 + 0]$
  - $y_i = \text{landmarks63}[i \times 3 + 1]$
  - $z_i = \text{landmarks63}[i \times 3 + 2]$

### 3.3 Sensor Rotation & CameraX Ingestion
Android camera sensors typically deliver frames in landscape orientation relative to natural device portrait orientation (requiring 90° or 270° counter-clockwise rotation):
1. **`ImageProxy` Ingestion:** The detector inspects `imageProxy.imageInfo.rotationDegrees` (0°, 90°, 180°, 270°).
2. **Zero-Copy Conversion:** Where available, `imageProxy.image` (`android.media.Image`) is packaged directly via `MediaImageBuilder(image).build()`. If unavailable, `imageProxy.toBitmap()` with `BitmapImageBuilder(bitmap).build()` serves as fallback.
3. **Orientation Forwarding:** `ImageProcessingOptions.builder().setRotationDegrees(rotationDegrees).build()` informs MediaPipe of the native sensor rotation, guaranteeing output landmarks are normalized to the oriented user view space $[0.0, 1.0]$.
4. **Ownership Discipline:** `detect(imageProxy)` processes the buffer without closing it, allowing the enclosing CameraX analyzer `finally { imageProxy.close() }` block to manage buffer return.

---

## 4. Mapping to `DetectedHandCandidate`

The conversion method `HandLandmarkDetector.convertResultToCandidates(result: HandLandmarkerResult?)` translates raw MediaPipe objects into domain models:

```kotlin
data class DetectedHandCandidate(
    val label: String,           // "Left" or "Right"
    val score: Float,            // Handedness confidence in [0.0, 1.0]
    val landmarks63: FloatArray, // 21 (x, y, z) normalized landmarks (size 63)
    val wristX: Float = landmarks63[0],
    val wristY: Float = landmarks63[1]
)
```

### Invariant Enforcements:
1. **Hand Clamping:** At most 2 hands are extracted (`minOf(allLandmarks.size, 2)`).
2. **Handedness Normalization:** Case-insensitive canonicalization mapping `"left"`/`"Left"` to `"Left"`, and `"right"`/`"Right"` to `"Right"`. Empty or missing categories default to `"Unknown"`.
3. **Score Clamping:** Category confidence is coerced into $[0.0, 1.0]$. Missing categories assign score `0.0f`.
4. **Wrist Anchor Consistency:** Explicit assignment `wristX = landmarks63[0]` and `wristY = landmarks63[1]` guarantees immediate spatial indexing for Hungarian bipartite matching.
5. **Robust Zero-Padding:** If fewer than 21 landmarks are provided by MediaPipe, missing dimensions are deterministic zeros, preserving exact length 63.
6. **Zero-Hand Safety:** Empty detection lists, null results, or missing frames return `emptyList<DetectedHandCandidate>()` cleanly without allocating intermediate collections.

---

## 5. Downstream Integration with `HandSlotResolver`

The `DetectedHandCandidate` list flows into `HandSlotResolver.resolve(candidates)`:
- **Slot 0:** Designated for Viewer-Left Hand (subject's right hand in selfie mode).
- **Slot 1:** Designated for Viewer-Right Hand (subject's left hand in selfie mode).
- **Hungarian Cost Matrix:** Resolves ambiguities when two hands are detected with identical handedness labels or cross-over trajectories using combined spatial wrist position ($x$) and classification confidence.
- **Verification:** Unit test `testDetectedHandCandidateIntegrationWithHandSlotResolver` validates end-to-end compatibility between `HandLandmarkDetector` candidates and `HandSlotResolver`.

---

## 6. Resource Lifecycle Management

- **`AutoCloseable` Contract:** `IHandLandmarkDetector` extends `java.lang.AutoCloseable`.
- **Deterministic Teardown:** Invoking `close()` tears down the underlying native MediaPipe pipeline:
  ```kotlin
  override fun close() {
      if (!_isClosed) {
          _isClosed = true
          try {
              engine.close()
          } catch (e: Exception) {
              Log.w(TAG, "Exception during detector close: ${e.message}", e)
          }
      }
  }
  ```
- **State Guarding:** Calling `detect()` after closure throws `IllegalStateException("HandLandmarkDetector has been closed")`.
- **Idempotency:** Repeated calls to `close()` are safe no-ops.

---

## 7. Unit Test Results

Unit tests were executed via Gradle (`testDebugUnitTest`) against `com.echogesture.app.vision.HandLandmarkDetectorTest`:

| # | Test Method | Scope / Invariant Verified | Result | Duration |
| :-: | :--- | :--- | :-: | :-: |
| 1 | `testContractConstants` | Verifies default asset path, confidence thresholds, and feature dimensions | **PASSED** | < 1 ms |
| 2 | `testConvertResultSingleRightHand` | Verifies 1-hand extraction, 63-D flattening, wrist mapping, pinky tip index | **PASSED** | < 1 ms |
| 3 | `testConvertResultTwoHands` | Verifies concurrent 2-hand extraction with independent labels and scores | **PASSED** | < 1 ms |
| 4 | `testHandednessNormalizationAndScoreClamping` | Verifies case normalization ("left" -> "Left") and confidence clamping [0.0, 1.0] | **PASSED** | < 1 ms |
| 5 | `testMissingHandednessCategoryFallback` | Verifies graceful fallback to label "Unknown" and score 0.0f when categories absent | **PASSED** | 1 ms |
| 6 | `testCoordinateBounds` | Verifies all 21 normalized landmarks (x, y) satisfy $0.0 \le coord \le 1.0$ | **PASSED** | 7 ms |
| 7 | `testZeroHandsAndNullResult` | Verifies empty list returned on 0 detected hands or null result | **PASSED** | 1 ms |
| 8 | `testClampingMoreThanTwoHands` | Verifies detector strictly clamps candidate output to maximum 2 hands | **PASSED** | 8 ms |
| 9 | `testPartialLandmarksZeroPadded` | Verifies partial detections (< 21 keypoints) are zero-padded to exact size 63 | **PASSED** | 22 ms |
| 10 | `testDetectorLifecycleAndClose` | Verifies initialized status, idempotent close(), and resource release flags | **PASSED** | 1 ms |
| 11 | `testDetectorTestEngineInjection` | Verifies desktop JVM engine injection and conversion fidelity | **PASSED** | 2 ms |
| 12 | `testDetectedHandCandidateIntegrationWithHandSlotResolver` | Verifies direct pipeline handshake with downstream Hungarian HandSlotResolver | **PASSED** | 1 ms |

**Total Vision Tests:** 12  
**Passing:** 12 (100%)  
**Failures:** 0  
**Errors:** 0  

---

## 8. File Inventory

| File Path | Description |
| :--- | :--- |
| `android/app/src/main/assets/hand_landmarker.task` | Frozen MediaPipe hand landmarker model bundle (7.82 MB, SHA256 verified) |
| `android/app/src/main/java/com/echogesture/app/vision/IHandLandmarkDetector.kt` | Interface decoupling caller from native vision runtime |
| `android/app/src/main/java/com/echogesture/app/vision/HandLandmarkDetector.kt` | Production implementation wrapping MediaPipe Tasks Vision with JVM fallback |
| `android/app/src/test/java/com/echogesture/app/vision/HandLandmarkDetectorTest.kt` | Comprehensive 12-test unit verification suite |
| `android/app/build.gradle.kts` | Configured `noCompress += "task"` for zero-decompression asset streaming |
| `reports/phase7c_mediapipe_integration.md` | Full engineering integration report and specifications |
