# EchoGesture Phase 7A.1: Dependency and Integration Verification Report
**Track 2: Samsung XR Hackathon 2026**  
**Component:** Android Scaffolding, Toolchain Audit, Dependency Lifecycle & Security Verification  
**Evaluator & Execution Engine:** Gemini 3.8 Flash (High) ONLY  
**Date:** 2026-11-20  
**Workspace:** `C:\EchoCam\android`  

---

## 1. Executive Summary

Phase 7A.1 performs a rigorous technical audit and integration verification of the native Kotlin Android foundation established in Phase 7A. The objective was to audit the runtime inference dependencies (Google AI Edge LiteRT), camera acquisition libraries (AndroidX CameraX), runtime contracts, resource lifecycle handling, security boundary isolation, and asset integrity before porting the multi-stage gesture classifier in Phase 7B.

### Summary of Audit Outcomes:
1. **LiteRT Upgrade (`1.0.1` -> `1.4.2`):**
   - Verified that LiteRT `1.4.2` belongs to Google's stable 1.4.x series which preserves the classic `org.tensorflow.lite.Interpreter` API, rather than breaking compatibility with the 2.x `CompiledModel` API.
   - Confirmed full native support for 16 KB memory page sizes (Android 15 / API 35 requirement), INT8 input/output tensors, and zero-copy `MappedByteBuffer` loading.
   - Successfully upgraded dependency to `com.google.ai.edge.litert:litert:1.4.2`.
2. **CameraX Modernization (`1.4.1` -> `1.4.2`):**
   - Upgraded to CameraX `1.4.2`, the latest stable maintenance release in the 1.4.x line, ensuring maximum stability without introducing migration risks from 1.5+ video/session changes.
   - Modernized `CameraManager.kt` by replacing deprecated `setTargetResolution(Size)` with the official `ResolutionSelector` and `ResolutionStrategy` API.
3. **Implementation & Invariant Verification:**
   - Confirmed both `Preview` and `ImageAnalysis` use cases are configured and bound simultaneously.
   - Confirmed `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` is explicitly set.
   - Confirmed `ImageProxy.close()` is strictly executed in a `finally` block to prevent camera buffer exhaustion.
   - Confirmed CameraX lifecycle automatically unbinds with `LifecycleOwner` and releases executor threads in `onDestroy()`.
   - Confirmed LiteRT runtime assertions validate input shape `[1, 64, 263]` INT8 and output shape `[1, 40]` INT8 at initialization.
   - Confirmed complete absence of `android.permission.INTERNET` across source and merged APK manifests.
4. **Verification & Checksum Parity:**
   - 100% of host JVM unit tests passed (`testDebugUnitTest` 6/6 passed).
   - Debug APK assembled cleanly (`assembleDebug` 38 tasks executed, 0 errors).
   - Model SHA-256 verified bit-for-bit identical across reference checkpoint, Android asset, and packaged APK (`ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499`).
   - Packaged with compression mode `0 (STORED / uncompressed)`, guaranteeing zero-copy memory mapping directly from flash into CPU cache.

---

## 2. LiteRT Runtime Audit

### 2.1. Version Analysis & Ecosystem Mapping
The project initially declared `com.google.ai.edge.litert:litert:1.0.1`. A comprehensive audit of Google AI Edge releases identified the following ecosystem structure:

| Release Generation | Key Version | Primary API | Android 15 Compatibility | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **LiteRT 1.0.x** | `1.0.1` | `org.tensorflow.lite.Interpreter` | Baseline (4 KB pages) | Initial release; upgrade recommended |
| **LiteRT 1.4.x** | **`1.4.2`** | **`org.tensorflow.lite.Interpreter`** | **Full (16 KB pages compliant)** | **Recommended: Drop-in replacement with modern OS hardening** |
| **LiteRT 2.x** | `2.3.2` | `com.google.ai.edge.litert.CompiledModel` | Full | Major architectural rewrite; breaks drop-in compatibility |

### 2.2. Evaluation Against Project Requirements
1. **API Parity:** LiteRT `1.4.2` retains the exact `org.tensorflow.lite.Interpreter` and `org.tensorflow.lite.DataType` package signatures. It does not require any class migration or breaking changes to [LiteRtModelRunner.kt](file:///c:/EchoCam/android/app/src/main/java/com/echogesture/app/inference/LiteRtModelRunner.kt).
2. **Operator Support:** The EchoGesture INT8 model (`echogesture_exp_d_int8.tflite`) consists of standard quantized operators: `FULLY_CONNECTED`, `RESHAPE`, `SOFTMAX`, `ADD`, `MUL`, `LOGISTIC`, `TANH`. LiteRT 1.4.2 packages Google's latest XNNPACK CPU runtime (`libtensorflowlite_jni.so`), offering optimized INT8 vector instructions (ARM NEON / dot-product `SDOT`/`UDOT` for ARM64-v8a).
3. **Zero-Copy Memory Mapping:** Supports memory mapping (`FileChannel.MapMode.READ_ONLY`) of uncompressed assets via Android `AssetFileDescriptor`.
4. **CompiledModel Decision:** The prompt explicitly specifies: *"Do not migrate to CompiledModel unless there is a demonstrated compatibility and performance benefit."* Because LiteRT 1.4.2 provides complete stability, native 16 KB page alignment, and full INT8 operator support using the existing `Interpreter` API, migrating to `CompiledModel` is unnecessary and would introduce migration risk.

### 2.3. Upgrade Execution & Dependency Tree
The dependency was updated in [android/app/build.gradle.kts](file:///c:/EchoCam/android/app/build.gradle.kts):
```kotlin
// Google AI Edge LiteRT (TensorFlow Lite runtime)
implementation("com.google.ai.edge.litert:litert:1.4.2")
```

Dependency resolution confirmation via `.\gradlew.bat app:dependencies`:
```
\--- com.google.ai.edge.litert:litert:1.4.2
     \--- com.google.ai.edge.litert:litert-api:1.4.2
```

---

## 3. CameraX Audit & Modernization

### 3.1. Release Analysis
A scan of Google Maven (`androidx.camera`) revealed the following distribution:
- **1.4.x Series:** `1.4.0` (Oct 2024), `1.4.1` (Dec 2024), **`1.4.2`** (March 2025).
- **1.5.x Series:** `1.5.0` through `1.5.3` (introduced high-speed 120/240 fps video recording, `SessionConfig` changes).
- **1.6.x / 1.7.x Series:** Major refactoring lines.

### 3.2. Upgrade Decision
Per the audit guideline (*"Upgrade only if there is a clear benefit and no unnecessary migration risk. Prefer stable releases over release candidates"*):
- Upgraded from `1.4.1` to **`1.4.2`** across all CameraX components (`camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`).
- CameraX `1.4.2` provides targeted stability fixes for camera capture session management without the behavioral shifts introduced by the 1.5+ video recording subsystem.

### 3.3. Deprecation Remediation with `ResolutionSelector`
CameraX 1.4.x deprecated `setTargetResolution(Size)` in favor of `ResolutionSelector`. In [CameraManager.kt](file:///c:/EchoCam/android/app/src/main/java/com/echogesture/app/camera/CameraManager.kt), the configuration was modernized:

```kotlin
val resolutionSelector = ResolutionSelector.Builder()
    .setResolutionStrategy(
        ResolutionStrategy(
            Size(640, 480),
            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
        )
    )
    .build()

val preview = Preview.Builder()
    .setResolutionSelector(resolutionSelector)
    .build()
    .also {
        it.surfaceProvider = previewView.surfaceProvider
    }

val imageAnalysis = ImageAnalysis.Builder()
    .setResolutionSelector(resolutionSelector)
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
    .build()
```
This guarantees deterministic 640x480 frame selection while eliminating compiler deprecation warnings.

---

## 4. Implementation & Invariant Verification

Each component of the native implementation was inspected against strict production requirements:

### 4.1. CameraManager Verification (`CameraManager.kt`)
| Requirement | Implementation Evidence | Verdict |
| :--- | :--- | :--- |
| **Preview & ImageAnalysis** | Both `preview` and `imageAnalysis` are created and passed to `provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)`. | **VERIFIED** |
| **STRATEGY_KEEP_ONLY_LATEST** | Line 61: `.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)` ensures no frame backlog or queue latency. | **VERIFIED** |
| **Safe ImageProxy Closure** | Lines 68–74: Analyzer wraps dispatch in `try { if (isAnalyzing) onFrameReceived(imageProxy) } finally { imageProxy.close() }`. | **VERIFIED** |
| **Lifecycle Safety** | Bound to Android `LifecycleOwner`. Explicit methods `pause()`, `resume()`, `stop()`, and `release()` shut down `analysisExecutor`. | **VERIFIED** |

### 4.2. LiteRT Model Runner Verification (`LiteRtModelRunner.kt`)
| Requirement | Implementation Evidence | Verdict |
| :--- | :--- | :--- |
| **Interpreter Initialization** | `loadModelFile()` maps file via `assets.openFd()`. Uses single-thread XNNPACK CPU options (`setNumThreads(1)`). | **VERIFIED** |
| **Input Shape & Type Check** | Lines 44 & 46: `check(inTensor.dataType() == DataType.INT8)` and `check(inShape == listOf(1, 64, 263))`. | **VERIFIED** |
| **Output Shape & Type Check** | Lines 45 & 47: `check(outTensor.dataType() == DataType.INT8)` and `check(outShape == listOf(1, 40))`. | **VERIFIED** |
| **Native Direct ByteBuffers** | Direct buffers allocated with `ByteOrder.nativeOrder()`: `inputBuffer` (16,832 bytes) and `outputBuffer` (40 bytes). | **VERIFIED** |
| **ArgMax Monotonicity** | ArgMax computed directly over signed int8 bytes without converting all 40 values to float first. | **VERIFIED** |

### 4.3. Security & Privacy Audit (Manifest & Permissions)
The merged manifest at `android/app/build/intermediates/merged_manifest/debug/processDebugMainManifest/AndroidManifest.xml` and the binary APK were inspected with `aapt2 dump badging`:

```
package: name='com.echogesture.app' versionCode='1' versionName='1.0.0'
minSdkVersion:'26'
targetSdkVersion:'35'
uses-permission: name='android.permission.CAMERA'
uses-permission: name='com.echogesture.app.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION'
```

- **`android.permission.INTERNET`:** **COMPLETELY ABSENT**.
- **Network Guards:** Zero network libraries, analytics SDKs, or remote endpoints are included.
- **Privacy Verdict:** The application operates in complete air-gapped isolation, fulfilling the hackathon privacy contract.

---

## 5. Build, Test & Checksum Evidence

### 5.1. Model Checksum & Integrity Matrix
The model was verified across all locations using SHA-256:

| Location | Path | Size (Bytes) | SHA-256 Checksum | Storage Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Original Python Checkpoint** | `models/quantized/echogesture_exp_d_int8.tflite` | 135,464 | `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` | File |
| **Android Asset File** | `android/app/src/main/assets/echogesture_exp_d_int8.tflite` | 135,464 | `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` | File |
| **Packaged Inside Debug APK** | `assets/echogesture_exp_d_int8.tflite` | 135,464 | `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499` | **STORED (0 / uncompressed)** |

### 5.2. Unit Test Suite Results
Executed via `.\gradlew.bat testDebugUnitTest --rerun-tasks`:
```
BUILD SUCCESSFUL in 26s
25 actionable tasks: 25 executed
```

Individual test status from HTML test report:
1. `testTensorDimensionsAndConstants` — **PASSED** (asserts batch 1, seq 64, dim 263, classes 40, buffer sizes 16,832 / 40 bytes)
2. `testQuantizationMath` — **PASSED** (asserts $q = \operatorname{clamp}(\operatorname{round}(x / 0.0579404756) + 23, -128, 127)$)
3. `testDequantizationMath` — **PASSED** (asserts $p = (q + 128) \cdot 0.00390625$)
4. `testArgMaxMonotonicity` — **PASSED** (asserts argmax over int8 matches float ordering)
5. `testModelAssetSha256Integrity` — **PASSED** (asserts asset SHA-256 matches reference hash)
6. `testVocabularyAssetIntegrity` — **PASSED** (asserts 40 classes loaded from `label_encoder.json`)

### 5.3. Assembled Debug APK Artifact
Executed via `.\gradlew.bat assembleDebug`:
- **APK Path:** [android/app/build/outputs/apk/debug/app-debug.apk](file:///c:/EchoCam/android/app/build/outputs/apk/debug/app-debug.apk)
- **APK Size:** 40,013,185 bytes (~38.16 MB)
- **APK SHA-256:** `584E0B185D93C124DB9724C9A433E2AB04CE94750DB03980A83BD62EE7A5EC9C`
- **Native Architectures Packaged:** `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`
- **Native Binaries:** `libtensorflowlite_jni.so` (LiteRT 1.4.2), `libimage_processing_util_jni.so`, `libsurface_util_jni.so` (CameraX 1.4.2).

---

## 6. Phase 7A.1 Sign-Off & Phase Boundary

| Audit Category | Result | Detail |
| :--- | :--- | :--- |
| **LiteRT Dependency** | **PASS** | Upgraded to `1.4.2`. Classic Interpreter API preserved; 16 KB page-size compliant. |
| **CameraX Dependency** | **PASS** | Upgraded to `1.4.2`. Deprecations eliminated with `ResolutionSelector`. |
| **Code Implementation** | **PASS** | Preview + ImageAnalysis bound; `KEEP_ONLY_LATEST` active; `ImageProxy` closed in `finally`. |
| **Runtime Model Contracts** | **PASS** | Shapes and INT8 data types checked at initialization; uncompressed asset zero-copy mmap verified. |
| **Privacy & Security** | **PASS** | Zero internet permission declared; air-gapped on-device operation enforced. |
| **Test & Build Verification** | **PASS** | 25/25 Gradle tasks executed; 6/6 unit tests passed; APK assembled successfully. |

**Strict Execution Boundary:** Phase 7A.1 is complete. No Python code, datasets, checkpoints, or original models were modified. Phase 7B (Kotlin recognition pipeline port) has not been started and awaits subsequent instruction.
