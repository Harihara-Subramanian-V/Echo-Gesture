# EchoGesture Production Inference Contract

**Project**: EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Document**: Production Inference Specification & Pipeline Contract  
**Model Artifact**: `models/quantized/echogesture_exp_d_int8.tflite` (132.29 KB)  
**Vocabulary**: 40 American Sign Language signs (MS-ASL canonical subset)  
**Target Hardware**: Mobile Edge / Samsung XR Headset (Snapdragon XR2 Gen 2 / Android)  

---

## 1. Overview & System Invariants

This document specifies the exact, mathematically verified contract required to process real-time camera frames into high-accuracy sign language predictions. Every transformation step matches the exact conventions established during Phase 3 feature engineering, Phase 4 sequence standardization, and Phase 5A/5B training and quantization.

```
+---------------------------------------------------------------------------------------------------+
|                                 END-TO-END INFERENCE PIPELINE STAGES                              |
+-------+----------------------------------+---------------------+----------------------------------+
| Stage | Operation                        | Input Shape         | Output Shape / Representation    |
+-------+----------------------------------+---------------------+----------------------------------+
| 1     | MediaPipe Hand Landmark Tracking | Live Video Frames   | 21 (x,y,z) landmarks per hand    |
| 2     | 134-D Frame Feature Assembly     | MediaPipe Detections| Variable length (T, 134) float32 |
| 3     | Absence-Aware Resampling (L=64)  | (T, 134) float32    | (64, 134) float32                |
| 4     | Active-Frame Standardization     | (64, 134) float32    | (64, 134) standardized float32   |
| 5     | Spike-Free Velocity Extraction   | (64, 134) float32    | 129-D velocity features          |
| 6     | Feature Vector Concatenation     | 134-D + 129-D       | (64, 263) float32                |
| 7     | Affine INT8 Input Quantization   | (64, 263) float32    | (1, 64, 263) int8                |
| 8     | TFLite INT8 Graph Execution      | (1, 64, 263) int8   | (1, 40) int8 logits              |
| 9     | Output Dequantization & Argmax   | (1, 40) int8        | Class name, ID, Top-5 Confidences|
+-------+----------------------------------+---------------------+----------------------------------+
```

---

## 2. Detailed Stage-by-Stage Transformations

### Stage 1: MediaPipe Hand Landmark Tracking
- **Detector**: MediaPipe Hands (`static_image_mode=False`, `max_num_hands=2`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`).
- **Slot Convention**:
  - Slot 0: Viewer-Left Hand (MediaPipe label `'Left'`).
  - Slot 1: Viewer-Right Hand (MediaPipe label `'Right'`).
- **Tracking Cache**: When both hands have identical handedness or cross in screen space, Hungarian nearest-wrist Euclidean assignment against frame $t-1$ maintains slot continuity.

### Stage 2: 134-Dimensional Frame Feature Assembly
For each video frame $t \in [0, T-1]$, assemble a 134-dimensional feature vector:
- `[0:63]`: Left Hand 21 normalized landmarks $(x, y, z)$ in viewer space $[0, 1]$.
- `[63:126]`: Right Hand 21 normalized landmarks $(x, y, z)$ in viewer space $[0, 1]$.
- `[126]`: Left hand binary presence flag ($1.0$ if detected, $0.0$ if absent).
- `[127]`: Right hand binary presence flag ($1.0$ if detected, $0.0$ if absent).
- `[128]`: Left hand handedness classification confidence score $p \in [0.0, 1.0]$.
- `[129]`: Right hand handedness classification confidence score $p \in [0.0, 1.0]$.
- `[130]`: Both hands co-present flag: $1.0$ if $p_L > 0.5 \land p_R > 0.5$, else $0.0$.
- `[131]`: Aspect-ratio corrected diagonal-normalized Euclidean inter-wrist distance $d_{\text{geom}} \in [0.0, 1.0]$:
  $$d_{\text{geom}} = \frac{\sqrt{((x_R - x_L) \cdot W)^2 + ((y_R - y_L) \cdot H)^2}}{\sqrt{W^2 + H^2}}$$
- `[132:134]`: Diagonal-normalized horizontal and vertical offsets $(\Delta x_{\text{geom}}, \Delta y_{\text{geom}}) \in [-1.0, 1.0]$:
  $$\Delta x_{\text{geom}} = \frac{(x_R - x_L) \cdot W}{\sqrt{W^2 + H^2}}, \quad \Delta y_{\text{geom}} = \frac{(y_R - y_L) \cdot H}{\sqrt{W^2 + H^2}}$$
- **Absence Rule**: If a hand is absent, its 63 coordinates and handedness score are strictly $0.0$. If both hands are not co-present, $d_{\text{geom}}$, $\Delta x_{\text{geom}}$, and $\Delta y_{\text{geom}}$ are strictly $0.0$.

### Stage 3: Temporal Resampling to Fixed Length $L = 64$
- **Target Time Grid**: $u_j = \frac{j}{63} \cdot (T - 1)$ for $j \in [0 \dots 63]$.
- **Presence Resampling**: Nearest-neighbor interpolation: $\text{pres}(u_j) = \text{pres}(\text{round}(u_j))$.
- **Trajectory Interpolation**: Landmark coordinates are interpolated **only between frames where the hand was actually detected** to prevent pulling coordinates to $(0,0,0)$.
- **Strict Zero Clamping**: For any resampled query $u$ where $\text{pres}(u) == 0.0$, all 63 coordinates are clamped to $0.0$.

### Stage 4: Active-Frame Z-Score Standardization
- Scaler parameters fitted strictly on the training partition (`data/processed/sequences_v3/scaler_params.npz`):
  - Left Hand: $\boldsymbol{\mu}_L \in \mathbb{R}^{63}, \boldsymbol{\sigma}_L \in \mathbb{R}^{63}$ (computed over active Left frames).
  - Right Hand: $\boldsymbol{\mu}_R \in \mathbb{R}^{63}, \boldsymbol{\sigma}_R \in \mathbb{R}^{63}$ (computed over active Right frames).
  - Interaction: $\boldsymbol{\mu}_{\text{inter}} \in \mathbb{R}^3, \boldsymbol{\sigma}_{\text{inter}} \in \mathbb{R}^3$ (computed over co-present frames).
- **Transformation**:
  $$x'_i(t) = \begin{cases} \frac{x_i(t) - \mu_i}{\sigma_i + 10^{-6}}, & \text{if hand is present at frame } t \\ 0.0, & \text{if hand is absent at frame } t \end{cases}$$
- Flags ($126, 127, 130$) and scores ($128, 129$) remain unscaled in $[0.0, 1.0]$.

### Stage 5 & 6: First-Order Velocity Features & Concatenation (263-D)
- **Transition Spike Prevention Rule**: Velocity is calculated **only when the hand is detected in both frame $t$ and frame $t-1$**:
  $$\mathbf{v}_L(t) = \begin{cases} \mathbf{c}_L(t) - \mathbf{c}_L(t-1), & \text{if } p_L(t) > 0.5 \land p_L(t-1) > 0.5 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
  $$\mathbf{v}_R(t) = \begin{cases} \mathbf{c}_R(t) - \mathbf{c}_R(t-1), & \text{if } p_R(t) > 0.5 \land p_R(t-1) > 0.5 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
  $$\mathbf{v}_{\text{inter}}(t) = \begin{cases} \mathbf{f}_{\text{inter}}(t) - \mathbf{f}_{\text{inter}}(t-1), & \text{if } p_{\text{both}}(t) > 0.5 \land p_{\text{both}}(t-1) > 0.5 \\ \mathbf{0}, & \text{otherwise} \end{cases}$$
  - Boundary: $\mathbf{v}(0) = \mathbf{0}$.
- **Vector Concatenation**:
  $$\mathbf{X}_{263} = [\mathbf{x}_{134}, \mathbf{v}_L(63), \mathbf{v}_R(63), \mathbf{v}_{\text{inter}}(3)] \in \mathbb{R}^{64 \times 263}$$

### Stage 7: Full-Integer Input Quantization
- **Parameters**:
  - Scale: $S_{\text{in}} = 0.05794025$
  - Zero Point: $Z_{\text{in}} = 23$
- **Formula**:
  $$q_{t, d} = \text{clip}\left(\text{round}\left(\frac{X_{t, d}}{S_{\text{in}}}\right) + Z_{\text{in}}, -128, 127\right) \in \text{int8}$$
  Shape passed to interpreter: `(1, 64, 263)` int8.

### Stage 8: TFLite INT8 Graph Execution
- Execute `interpreter.invoke()`.
- Model output: tensor of shape `(1, 40)` int8.

### Stage 9: Output Interpretation & Dequantization
- **Parameters**:
  - Scale: $S_{\text{out}} = 0.00390625$ ($1 / 256$)
  - Zero Point: $Z_{\text{out}} = -128$
- **Formula**:
  $$p_k = (q_{\text{out}, k} - Z_{\text{out}}) \times S_{\text{out}} = (q_{\text{out}, k} + 128) / 256.0 \in [0.0, 1.0]$$
- **Predicted Class**:
  $$\hat{k} = \text{argmax}_{k}(q_{\text{out}, k})$$
  (Since quantization is an affine monotonic transform, $\text{argmax}(q) \equiv \text{argmax}(p)$).
- **Top-5 Ranking**: Sort classes by $q_{\text{out}}$ descending and return top 5 entries with decoded class names from `label_encoder.json`.

---

## 3. Vocabulary Mapping Reference (40 Classes)

The output logits correspond to the 40 sign classes sorted alphabetically:

| ID | Class Name | ID | Class Name | ID | Class Name | ID | Class Name |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| **0** | `all` | **10** | `deaf` | **20** | `like` | **30** | `table` |
| **1** | `bed` | **11** | `dog` | **21** | `man` | **31** | `teacher` |
| **2** | `before` | **12** | `drink` | **22** | `many` | **32** | `thanksgiving` |
| **3** | `black` | **13** | `family` | **23** | `mother` | **33** | `thin` |
| **4** | `blue` | **14** | `fine` | **24** | `no` | **34** | `walk` |
| **5** | `book` | **15** | `finish` | **25** | `now` | **35** | `water` |
| **6** | `bowling` | **16** | `fish` | **26** | `orange` | **36** | `white` |
| **7** | `can` | **17** | `girl` | **27** | `shirt` | **37** | `who` |
| **8** | `clothes` | **18** | `go` | **28** | `shoes` | **38** | `woman` |
| **9** | `computer` | **19** | `graduate` | **29** | `study` | **39** | `year` |

---

## 4. Verification Checkpoints

Any mobile or edge client implementing this contract must pass these verification invariants:
1. When input is all zeros, output must not crash or trigger NaN.
2. Input tensor must be exactly `int8` of shape `[1, 64, 263]`.
3. Predictions on the validation suite must match `reports/int8_quantization_evaluation.json` with 100% agreement.
