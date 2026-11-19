# EchoGesture System Architecture

**Document Version:** 0.1.0  
**Date:** 2026-11-19  
**Status:** Architecture Specification  
**Project:** EchoCam / EchoGesture  

---

## 1. Document Overview

This document specifies the end-to-end system architecture for **EchoGesture**, an on-device sign language recognition system designed for low-latency edge deployment on mobile hardware. 

---

## 2. System Overview

EchoGesture is engineered specifically for **limited-vocabulary isolated sign recognition** targeting approximately 40 discrete lexical classes (e.g., emergency commands, navigational directives, and essential communicative signs).

### Architectural Boundaries & Scope
- **Isolated Sign Recognition:** EchoGesture operates on temporally bounded, pre-segmented video clips containing a single executed sign gesture. It maps a sequence of motion frames to a single categorical prediction from a pre-defined lexicon.
- **Lexical Vocabulary Target:** Approximately 40 high-utility sign classes (e.g., American Sign Language isolated signs from standardized benchmarks such as WLASL).
- **NOT Unrestricted Translation:** EchoGesture is explicitly **not** a continuous sign-to-speech translation system, nor does it perform natural language syntax parsing, continuous sentence transcription, or open-vocabulary linguistic translation. Continuous sign language incorporates complex non-manual markers, spatial grammar, grammatical inflections, and co-articulation that are beyond the scope of this isolated recognition architecture.
- **Privacy-First & On-Device:** All sensory processing, computer vision extraction, feature computation, and neural network inference occur 100% locally on the host device without offloading frames or embeddings to cloud infrastructure.

---

## 3. Pipeline Architecture

The EchoGesture processing pipeline decomposes sign recognition into decoupled stages: spatial visual landmark extraction, geometric normalization, kinematic temporal modeling, and categorical classification.

```
+-----------------------------------------------------------------------------+
|                               VIDEO INPUT                                   |
|               (Local Camera Stream or Stored Video File)                   |
+-------------------------------------+---------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                              FRAME SAMPLING                                 |
|             (Uniform Temporal Sampling -> Exactly 30 Frames)                |
|                    (Spatial Resize: 224 x 224 RGB)                          |
+-------------------------------------+---------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                           MEDIAPIPE LANDMARKS                               |
|        (Left Hand: 21x3 | Right Hand: 21x3 | Pose Selected: 7x3)            |
|                 (Visibility / Presence Scores: 49)                          |
|                       (~196 Raw Values / Frame)                             |
+-------------------------------------+---------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                           FEATURE ENGINEERING                               |
|      (Translation Norm: Mid-Shoulder | Scale Norm: Shoulder Width)          |
|      (Temporal Velocities: Delta Coordinates | Missing Hand Masks)          |
|                        (~200 - 300 Dim / Frame)                             |
+-------------------------------------+---------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                             TEMPORAL MODEL                                  |
|         (Option A: 2-Layer GRU  |  Option B: 3-Stage Temporal CNN)          |
|                   (Input Tensor: [Batch, 30, Features])                     |
+-------------------------------------+---------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                               PREDICTION                                    |
|              (Softmax Distribution over ~40 Sign Classes)                   |
|               (Top-1 Class + Confidence Score + Top-5 Array)                |
+-----------------------------------------------------------------------------+
```

---

## 4. Component Details

### 4.1. Video Input
- **Source Modalities:** Real-time live camera capture via camera sensor API (Android CameraX / NDK Camera2) or recorded digital video files (`.mp4`, `.avi`, `.mov`).
- **Framing & Viewpoint:** Exocentric viewpoint (third-person frontal or slight three-quarter perspective). The camera frames the signer from the upper waist to the top of the head, capturing complete bilateral arm articulation and facial/torso anchor points.
- **Locality:** Fully local pipeline. Frame buffers reside exclusively in volatile application memory; no video streams or image data are transmitted across local networks or cloud APIs.

### 4.2. Frame Sampling
- **Temporal Strategy:** Uniform temporal sampling across the entire clip duration $T$. If a video contains $N$ total frames, indices are selected uniformly:
  $$i_k = \left\lfloor k \cdot \frac{N - 1}{29} \right\rfloor \quad \text{for } k \in [0, 29]$$
  This produces a fixed temporal window of exactly $T_{\text{fixed}} = 30$ frames regardless of native clip framerate (e.g., 24 FPS, 30 FPS, or 60 FPS) or gesture duration (1.0 to 4.0 seconds).
- **Spatial Preprocessing:** Selected frames are decoded, converted from native colorspace (e.g., YUV/BGR) to standard RGB, and scaled to a uniform spatial resolution of $224 \times 224$ pixels. This resolution optimizes MediaPipe detector throughput while preserving finger joint articulation.

### 4.3. MediaPipe Landmarks
Spatial tracking extracts geometric skeletons from each sampled frame using Google MediaPipe Tasks (Hands and Pose):
- **Left Hand Landmarks:** 21 anatomical landmarks per hand. Each landmark provides $(x, y, z)$ coordinates:
  $$21 \times 3 = 63 \text{ values}$$
- **Right Hand Landmarks:** 21 anatomical landmarks per hand, each providing $(x, y, z)$ coordinates:
  $$21 \times 3 = 63 \text{ values}$$
- **Selected Pose Landmarks:** 7 key upper-body landmarks selected from the standard 33-point Pose model to anchor gestures to torso anatomy:
  1. Nose (`landmark 0`)
  2. Left Shoulder (`landmark 11`)
  3. Right Shoulder (`landmark 12`)
  4. Left Elbow (`landmark 13`)
  5. Right Elbow (`landmark 14`)
  6. Left Wrist (`landmark 15`)
  7. Right Wrist (`landmark 16`)
  $$7 \times 3 = 21 \text{ values}$$
- **Visibility / Presence Scores:** Confidence indicators representing the visibility or presence probability for each tracked landmark:
  - 21 values (Left Hand) + 21 values (Right Hand) + 7 values (Pose joints) = 49 values.
- **Raw Landmark Vector:**
  $$\text{Total per frame} = 63 + 63 + 21 + 49 = 196 \text{ values}$$
  Across 30 sampled frames, the raw tensor representation is $[30, 196]$.

### 4.4. Feature Engineering
Raw screen-space pixel coordinates are sensitive to subject positioning, camera distance, and body size. The feature engineering module applies deterministic spatial and kinematic transformations:

1. **Translation Invariance (Anchor Normalization):**
   The midpoint between the left shoulder $P_{\text{l\_shoulder}}$ and right shoulder $P_{\text{r\_shoulder}}$ serves as the anatomical origin $O_{\text{body}}$:
   $$O_{\text{body}} = \frac{P_{\text{l\_shoulder}} + P_{\text{r\_shoulder}}}{2}$$
   All 3D coordinates $(x, y, z)$ for hands and pose are translated relative to $O_{\text{body}}$:
   $$P_{\text{rel}} = P - O_{\text{body}}$$
   If shoulders are not detected, the pose nose joint or neck center acts as secondary fallback origin.

2. **Scale Invariance (Shoulder-Width Scaling):**
   The Euclidean distance between the left and right shoulders defines the reference scale factor $S_{\text{body}}$:
   $$S_{\text{body}} = \|P_{\text{l\_shoulder}} - P_{\text{r\_shoulder}}\|_2$$
   All translated coordinates are normalized by $S_{\text{body}}$:
   $$P_{\text{norm}} = \frac{P_{\text{rel}}}{S_{\text{body}} + \epsilon}$$
   where $\epsilon = 10^{-6}$ prevents division by zero. This normalizes for differences in signer physical stature and variable distance from the camera lens.

3. **Kinematic Velocity Features:**
   Dynamic gestures are distinguished by movement speed and trajectory. First-order temporal backward differences are computed across consecutive frames:
   $$\Delta P_{\text{norm}}(t) = P_{\text{norm}}(t) - P_{\text{norm}}(t - 1) \quad \text{for } t \in [1, 29]$$
   For the initial frame ($t = 0$), $\Delta P_{\text{norm}}(0) = 0$.

4. **Occlusion & Missing Landmark Masks:**
   Sign language frequently involves one-handed gestures or momentary hand occlusions. Binary indicator masks (0 = absent/occluded, 1 = tracked) are concatenated with the feature vector. Missing joint positions are imputed with zero vectors to avoid NaN propagation.

5. **Composite Feature Vector:**
   The concatenated frame descriptor combines normalized coordinates (49 joints $\times$ 3 = 147), kinematic velocities (147), and detection presence masks (49), yielding an engineered feature vector of **200 to 300 dimensions** per frame (typically 245 to 294 continuous values depending on velocity pruning). The sequence input tensor fed to the temporal model has fixed shape $[30, D_{\text{feat}}]$ (e.g., $[30, 245]$).

### 4.5. Temporal Model Options
Two alternative lightweight sequence backbones are evaluated for on-device inference:

```
Option A: Stacked Recurrent (GRU)         Option B: Temporal Convolutional (TCNN)
+-------------------------------+         +-------------------------------------+
| Input Sequence: [30, D_feat]  |         | Input Sequence: [30, D_feat]        |
+---------------+---------------+         +------------------+------------------+
                |                                            |
                v                                            v
+-------------------------------+         +-------------------------------------+
| Layer 1: GRU (128 units)      |         | Block 1: Conv1D (64 filters, d=1)   |
| (Recurrent Dropout = 0.2)     |         | (BatchNorm + ReLU + Residual)       |
+---------------+---------------+         +------------------+------------------+
                |                                            |
                v                                            v
+-------------------------------+         +-------------------------------------+
| Layer 2: GRU (64 units)       |         | Block 2: Conv1D (128 filters, d=2)  |
| (Last Step Output: 64 dim)    |         | (BatchNorm + ReLU + Residual)       |
+---------------+---------------+         +------------------+------------------+
                |                                            |
                v                                            v
+-------------------------------+         +-------------------------------------+
| Dense Projection (64 -> 64)   |         | Block 3: Conv1D (64 filters, d=4)   |
| (Dropout = 0.3, ReLU)         |         | (BatchNorm + ReLU + Residual)       |
+---------------+---------------+         +------------------+------------------+
                |                                            |
                v                                            v
+-------------------------------+         +-------------------------------------+
| Dense Classification Head     |         | Global Average Pooling 1D           |
| (Dense 40, Softmax)           |         +------------------+------------------+
+-------------------------------+                            |
                                                             v
                                          +-------------------------------------+
                                          | Dense Classification Head           |
                                          | (Dense 40, Softmax)                 |
                                          +-------------------------------------+
```

#### Specification Comparison

| Architectural Dimension | Option A: Recurrent GRU | Option B: Temporal CNN (TCNN) |
| :--- | :--- | :--- |
| **Layer Configuration** | 2-Layer stacked GRU (128 $\to$ 64 units) | 3 $\times$ Conv1D blocks (64/128/64 filters, kernels 3/5, dilations 1, 2, 4) |
| **Receptive Field** | Full sequence via recursive hidden state | 15--29 frames via dilated convolutions |
| **Parameter Count** | ~200,000 -- 400,000 parameters | ~100,000 -- 300,000 parameters |
| **FP32 Model Size** | ~1.2 -- 2.4 MB | ~0.6 -- 1.8 MB |
| **INT8 Quantized Size** | **~1.0 -- 2.0 MB** | **~0.5 -- 1.5 MB** |
| **Execution Strengths** | Long-range context preservation; adaptive temporal dynamics | Constant latency; zero recurrent state; highly parallelizable on mobile NPUs/GPUs |

### 4.6. Prediction Output Schema
The temporal classification model outputs a probability distribution over the target classes via Softmax. Downstream application modules consume a structured JSON response:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EchoGesturePrediction",
  "type": "object",
  "properties": {
    "status": { "type": "string", "enum": ["SUCCESS", "NO_SIGNER_DETECTED", "LOW_CONFIDENCE"] },
    "timestamp_ms": { "type": "integer" },
    "inference_latency_ms": { "type": "number" },
    "top_prediction": {
      "type": "object",
      "properties": {
        "class_id": { "type": "integer" },
        "label": { "type": "string" },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
      },
      "required": ["class_id", "label", "confidence"]
    },
    "top_k_predictions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "rank": { "type": "integer" },
          "class_id": { "type": "integer" },
          "label": { "type": "string" },
          "confidence": { "type": "number" }
        },
        "required": ["rank", "class_id", "label", "confidence"]
      }
    }
  },
  "required": ["status", "timestamp_ms", "inference_latency_ms", "top_prediction", "top_k_predictions"]
}
```

#### Example Output Payload
```json
{
  "status": "SUCCESS",
  "timestamp_ms": 1732012641000,
  "inference_latency_ms": 42.8,
  "top_prediction": {
    "class_id": 14,
    "label": "HELP",
    "confidence": 0.9412
  },
  "top_k_predictions": [
    { "rank": 1, "class_id": 14, "label": "HELP", "confidence": 0.9412 },
    { "rank": 2, "class_id": 3,  "label": "EMERGENCY", "confidence": 0.0381 },
    { "rank": 3, "class_id": 22, "label": "STOP", "confidence": 0.0114 },
    { "rank": 4, "class_id": 7,  "label": "DOCTOR", "confidence": 0.0053 },
    { "rank": 5, "class_id": 31, "label": "WATER", "confidence": 0.0021 }
  ]
}
```

---

## 5. Latency Budget

To maintain seamless, real-time interactivity on modern smartphone processors without causing thermal throttling or UI frame drops, EchoGesture establishes the following stage-by-stage latency budget targets.

> [!NOTE]
> Values listed below represent design allocation targets for real-time responsiveness (<60 ms total pipeline turnaround), not empirical device benchmark measurements.

| Pipeline Stage | Processing Scope | Target Latency Budget | Budget Share |
| :--- | :--- | :---: | :---: |
| **1. Frame Extraction & Decoding** | Uniform 30-frame temporal indexing, hardware decoding, $224 \times 224$ downsampling | **5 -- 10 ms** | ~14% |
| **2. MediaPipe Landmark Extraction** | Hands (bilateral) + selected Pose joint detection across batch/frames | **15 -- 25 ms** | ~43% |
| **3. Feature Engineering** | Body-relative translation, shoulder-width scaling, velocities, masking | **2 -- 5 ms** | ~7% |
| **4. Sequence Model Inference** | Quantized TFLite INT8 GRU / TCNN execution on mobile NPU/GPU/CPU | **10 -- 20 ms** | ~36% |
| **Total End-to-End Latency** | **Camera frame ingestion to JSON prediction output** | **~35 -- 60 ms** | **100%** |

At an end-to-end turnaround of ~35--60 ms, EchoGesture achieves an operational throughput of 16 to 28 gesture evaluations per second, enabling near-instantaneous feedback upon completion of a sign motion.

---

## 6. Edge Deployment Architecture

EchoGesture is engineered to execute entirely self-contained on Android mobile devices. The architecture requires no external server calls, guaranteeing privacy, zero network latency, and operational availability in offline or remote environments.

```
+-----------------------------------------------------------------------------+
|                          ANDROID DEVICE BOUNDARY                            |
|                                                                             |
|  +--------------------+                                                     |
|  |   Camera Sensor    |                                                     |
|  | (CameraX / NDK)    |                                                     |
|  +---------+----------+                                                     |
|            | Native NV21 / ImageProxy Buffer                                |
|            v                                                                |
|  +--------------------+                                                     |
|  | MediaPipe Tasks    |  <-- Google MediaPipe Android AAR (C++ / Java Core) |
|  | (Hands + Pose)     |                                                     |
|  +---------+----------+                                                     |
|            | Normalized Joint Skeletons (196 floats / frame)                |
|            v                                                                |
|  +--------------------+                                                     |
|  | Feature Engine     |  <-- C++ / Kotlin Native Math Module                |
|  | (Normalize / Diff) |                                                     |
|  +---------+----------+                                                     |
|            | Model Input Tensor: [1, 30, Features] Float32 / INT8           |
|            v                                                                |
|  +--------------------+                                                     |
|  | TFLite Interpreter |  <-- TensorFlow Lite Runtime C++ Engine             |
|  | (INT8 Quantized)   |  <-- Hardware Acceleration: NNAPI / XNNPACK / GPU   |
|  +---------+----------+                                                     |
|            | Output Tensor: [1, 40] Softmax Probabilities                   |
|            v                                                                |
|  +--------------------+                                                     |
|  | UI Presentation    |  <-- Jetpack Compose UI View / Android TTS Engine   |
|  | (Text + Audio TTS) |                                                     |
|  +--------------------+                                                     |
|                                                                             |
|  =========================================================================  |
|  [AIR-GAPPED RUNTIME]: ZERO CLOUD DEPENDENCY | STRICT LOCAL PROCESSING     |
+-----------------------------------------------------------------------------+
```

### Mobile Execution Characteristics
- **Zero Cloud Network Dependency:** Operates fully without WiFi or cellular data.
- **Hardware Acceleration:** Uses TensorFlow Lite hardware delegates (NNAPI for Android Neural Networks API, XNNPACK for multi-threaded SIMD CPU execution, or OpenCL/OpenGL GPU delegates).
- **Memory Footprint:** Resident runtime memory stays well below 60 MB (MediaPipe pipeline ~35 MB, TFLite runtime + model ~10 MB, UI and buffers ~15 MB).

---

## 7. Technology Stack

The components spanning training, optimization, and edge inference are listed below:

| Architectural Tier | Component / Library | Version / Standard | Functional Purpose |
| :--- | :--- | :--- | :--- |
| **Development & Training** | Python | 3.10+ | Development environment and pipeline orchestration |
| **Vision & Video I/O** | OpenCV (`cv2`) | 4.10.0+ | Video decoding, resizing, color conversion (BGR to RGB) |
| **Skeletal Tracking** | Google MediaPipe | 0.10.14+ | Real-time hand and pose landmark extraction |
| **Array Computing** | NumPy | 1.26.4+ | Coordinate array manipulation and kinematic math |
| **Deep Learning Framework** | TensorFlow / Keras | 2.16+ / 3.x | Sequence model architecture, training loops, export |
| **Metrics & Evaluation** | scikit-learn | 1.4+ | Confusion matrices, Top-1/Top-5 accuracy, classification reports |
| **Model Optimization** | TFLite Converter | Built-in TF 2.16+ | Post-training INT8 / Float16 quantization |
| **Edge Mobile Runtime** | TensorFlow Lite | 2.16+ / Task Library | Low-latency on-device inference execution |
| **Mobile Acceleration** | XNNPACK / NNAPI | Android NDK Delegate | Hardware acceleration on mobile CPUs and NPUs |
| **Mobile Application Framework**| Kotlin / Jetpack Compose | Modern Android SDK (API 31+) | Camera lifecycle management and user interface |

---

## 8. Data Flow Architecture

The data flow diagram below traces the structural representations and tensor dimensionality transitions as raw video pixels are transformed into classification probabilities:

```
[Raw Video Stream]
  |
  |  File or Live Camera: [Variable Frames, Height, Width, 3 (BGR/YUV)]
  v
[Temporal Uniform Sampling & Resize]
  |
  |  Fixed Spatial-Temporal Tensor: [30, 224, 224, 3] (RGB uint8)
  v
[MediaPipe Landmark Inference]
  |
  |  Per Frame:
  |    Left Hand:   [21, 3] (x, y, z)
  |    Right Hand:  [21, 3] (x, y, z)
  |    Pose Joints: [7, 3]  (x, y, z)
  |    Visibility:  [49]    (v_score)
  |  Concatenated Raw Sequence: [30, 196] (float32)
  v
[Geometric Normalization & Kinematics]
  |
  |  1. Translation: P_norm = (P - O_body) / S_body
  |  2. Kinematics:  Delta_P = P_norm(t) - P_norm(t-1)
  |  3. Presence Masking & Concatenation
  |  Engineered Feature Tensor: [1, 30, D_feat] (float32, D_feat in [200, 300])
  v
[TFLite Sequence Classifier (INT8 / FP32)]
  |
  |  GRU Layers (128 -> 64) OR Temporal CNN (Dilated Conv1D Blocks)
  |  Dense Linear Head + Softmax Activation
  |  Logits / Probability Tensor: [1, 40] (float32, sum = 1.0)
  v
[Prediction Formatting & UI Dispatch]
  |
  |  Argmax -> Top-1 Class Label, Confidence Score
  |  Top-K Sorting -> Top-5 Candidate Ranking
  v
[Output Prediction JSON / TTS Speech Synthesis]
```

---

## 9. Limitations & Boundary Conditions

To ensure transparent operational expectations, EchoGesture adheres to the following explicit boundary constraints:

1. **Limited Vocabulary Scale (~40 Classes):**  
   EchoGesture is bounded to a closed vocabulary of approximately 40 discrete sign gestures. It cannot recognize, classify, or extrapolate words, compound concepts, or phrases outside its trained lexicon.
2. **Strictly Isolated Sign Format:**  
   The system requires temporally isolated signs with clear resting or transition boundaries. It cannot segment continuous signing streams or recognize fluent sign discourse without external continuous segmentation models.
3. **Language & Dialect Specificity (ASL Exclusivity):**  
   Training distributions are grounded exclusively in American Sign Language (ASL). It cannot parse British Sign Language (BSL), French Sign Language (LSF), or regional vernacular dialects due to fundamentally different phonological and lexical conventions.
4. **Single-Signer Constraint:**  
   The pipeline assumes a single primary signer within the camera's bounding perspective. If multiple individuals enter the frame, landmark tracking estimators experience cross-identity jitter and feature mingling, invalidating downstream inference.
5. **Landmark Extraction Dependency:**  
   The classification pipeline depends entirely on the spatial integrity of MediaPipe landmark coordinates. Severe joint occlusions, extreme motion blur, poor illumination (<50 lux), or hands leaving the frame will degrade recognition accuracy.
6. **No Fingerspelling Recognition:**  
   EchoGesture recognizes holistic, dynamic word-level gestures. It is not designed to parse rapid fingerspelling sequences (manual alphabet A--Z), which require high-frequency micro-finger articulation tracking.
7. **Absence of Non-Manual Grammatical Markers:**  
   Linguistic elements conveyed through facial expressions, mouth morphemes, eyebrow inflection, and head tilt are not parsed by the model, even though they represent essential grammatical indicators in natural sign language.
8. **Viewpoint Sensitivity & Exocentric Assumption:**  
   The architecture assumes a frontal or slight three-quarter exocentric perspective at a distance of 1.0 to 2.5 meters. Recognition will fail on egocentric (wearable / first-person) viewpoints or extreme vertical/lateral viewing angles.

---
*End of EchoGesture System Architecture Document.*
