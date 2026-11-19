# EchoGesture Phase 6B: Temporal Issues and Engineering Fixes Audit
**Component:** Live Inference Temporal Buffer, Stabilization & Hysteresis Logic  
**Date:** 2026-11-20  
**Evaluator:** Gemini 3.8 Flash (High)  

---

## 1. Executive Summary

During the Phase 6B audit, four concrete temporal reliability and streaming behavior issues were diagnosed in the initial live inference pipeline ([scripts/live_inference.py](file:///c:/EchoCam/scripts/live_inference.py)). All identified issues stemmed entirely from online streaming buffer management and temporal hysteresis filters, not the underlying neural network.

In accordance with strict execution rules, **minimal, targeted software corrections** were implemented directly in [scripts/live_inference.py](file:///c:/EchoCam/scripts/live_inference.py). The trained model ([models/quantized/echogesture_exp_d_int8.tflite](file:///c:/EchoCam/models/quantized/echogesture_exp_d_int8.tflite)), 134-D feature contract, 263-D velocity layout, and test splits remained 100% untouched.

Empirical re-evaluation across 40 representative held-out test clips demonstrated:
- **Missing Prediction Rate:** Reduced from **50.0% (20/40)** down to **2.5% (1/40)**.
- **Short Clip (< 64 frames) Failures:** Reduced from **100.0% missing (18/18)** down to **0.0% missing (0/18)**.
- **Median Time to First Stable Prediction:** Dropped from **64.0 frames (2.13s)** down to **17.0 frames (0.57s)**.
- **Overall Pipeline Accuracy:** Increased from **12.5% (5/40)** up to **30.0% (12/40)** without retraining.

---

## 2. Comprehensive Diagnosis of Issues & Applied Fixes

### Issue 1: Incomplete Window Failure on Short Clips & Initial Startup
- **Root Cause:**
  In the baseline pipeline, model execution was gated behind `if buf_len == 64:`. In MS-ASL, 49.7% of all test clips have fewer than 64 frames (minimum 22 frames). Consequently, any standalone gesture clip under 64 frames never filled the buffer and produced 0 predictions (100% missing). In live webcam use, signers performing quick 1-second gestures during the initial 2.13 seconds were completely ignored.
- **Minimal Correction Applied:**
  Lowered the inference trigger threshold from `buf_len == 64` to `buf_len >= 16` whenever `active_frames >= 12`. When $16 \le \text{buf\_len} < 64$, the available $K$ frames are dynamically resampled to length 64 using `scripts/prepare_sequences.resample_sequence` (linear 1D coordinate and geometry interpolation), identical to the offline training preparation. When $\text{buf\_len} == 64$, standard FIFO sliding is used without resampling overhead.
- **Verification:**
  Tested in `test_11_incomplete_window_resampling`. Missing short clips dropped from 18/18 (100%) to 0/18 (0%).

---

### Issue 2: Stale Prediction Persistence When Signing Concludes with Hands Visible
- **Root Cause:**
  In `PredictionStabilizer`, when the signer finished signing but left their hands resting or static in the frame, velocities dropped to $0.0$, causing model confidences to drop below the 0.35 threshold into `"UNCERTAIN"`. However, the stabilizer's weighted voting logic simply skipped `"UNCERTAIN"` frames without penalizing the current prediction. When all 5 frames in the window became `"UNCERTAIN"`, `class_scores` was empty, and the stabilizer returned the previous `current_stabilized_class` indefinitely. The last sign remained permanently frozen on screen.
- **Minimal Correction Applied:**
  Updated `PredictionStabilizer.update()`: when `class_scores` is empty (all frames in the sliding history window are `"UNCERTAIN"`), if the window has completely filled with uncertain frames (`len(self.history) >= self.window_size`), the stabilizer immediately resets to `"IDLE"`, confidence `0.0`, and ID `-1`.
- **Verification:**
  Tested in `test_12_stale_prediction_uncertain_decay`. Stale predictions now cleanly decay to `IDLE` after 5 consecutive uncertain frames (~166 ms at 30 FPS) while hands remain visible.

---

### Issue 3: Single-Frame Spike Vulnerability in Class Switching
- **Root Cause:**
  The baseline switching condition was `best_score >= self.confidence_threshold * 1.5` ($0.525$). A single transient model prediction with confidence 0.55 produced `best_score = 0.55 >= 0.525` and immediately flipped `current_stabilized_class` on that single frame, causing visual flicker when spurious classification spikes occurred.
- **Minimal Correction Applied:**
  Introduced asymmetrical hysteresis in `PredictionStabilizer.update()`:
  - Maintaining an *existing* stabilized class only requires `best_score >= self.confidence_threshold` ($0.35$).
  - Switching to a *new* class requires:
    1. Accumulated score $\ge 1.5 \times \text{threshold}$ ($0.525$), **AND**
    2. At least 2 confirming votes (`votes_count >= 2`) **OR** a dominant mass $\ge 2.0 \times \text{threshold}$ ($0.70$).
  Single-frame moderate noise spikes (e.g. 0.40–0.65) cannot hijack the UI without confirmation.
- **Verification:**
  Tested in `test_13_two_frame_consensus_hysteresis`. Single-frame spikes are suppressed while sustained transitions switch smoothly.

---

### Issue 4: Cross-Sign Temporal Window Contamination
- **Root Cause:**
  When a user completed Sign A, dropped their hands, and then initiated Sign B, the 64-frame FIFO buffer still retained the trailing frames of Sign A and the zero-padded drop frames, contaminating the first half of Sign B's temporal window.
- **Minimal Correction Applied:**
  In `run_live_inference()` and `evaluate_single_clip()`, whenever the stabilizer triggers idle dropout ($\ge 12$ consecutive idle frames), the temporal buffer is marked stale. Upon the very next hand detection onset, `buffer_134.clear()` is called, ensuring the new sign starts with a clean buffer primed exclusively by the new gesture.

---

## 3. Quantitative Before-and-After Comparison

Evaluation performed across the 40 representative held-out MS-ASL test clips:

| Metric | Baseline Pipeline (Before) | Improved Pipeline (After) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Clips Evaluated** | 40 | 40 | — |
| **Missing Predictions (Total)** | 20 (50.0%) | 1 (2.5%) | **-47.5% (-19 clips)** |
| **Short Clips (< 64f) Missing** | 18 / 18 (100.0%) | 0 / 18 (0.0%) | **-100.0% (-18 clips)** |
| **Long Clips ($\ge$ 64f) Missing** | 2 / 22 (9.1%) | 1 / 22 (4.5%) | **-4.6% (-1 clip)** |
| **Overall Top-1 Correct** | 5 / 40 (12.5%) | 12 / 40 (30.0%) | **+17.5% (+7 clips)** |
| **Short Clips Correct** | 0 / 18 (0.0%) | 7 / 18 (38.9%) | **+38.9% (+7 clips)** |
| **Long Clips Correct** | 5 / 22 (22.7%) | 5 / 22 (22.7%) | Preserved (no regression) |
| **Mean Time to First Prediction** | 64.8 frames (2.16s) | 24.1 frames (0.80s) | **-1.36s (-62.8%)** |
| **Median Time to First Prediction**| 64.0 frames (2.13s) | 17.0 frames (0.57s) | **-1.56s (-73.4%)** |

---

## 4. Architectural Non-Negotiables & Invariants Preserved

1. **Model Weights & Graph:** Unchanged. The pure INT8 TFLite model ([echogesture_exp_d_int8.tflite](file:///c:/EchoCam/models/quantized/echogesture_exp_d_int8.tflite)) was executed without retraining or recalibration.
2. **Feature Contracts:** 134-D canonical vector and 263-D preprocessed tensor schemas match [production_inference_contract.md](file:///c:/EchoCam/reports/production_inference_contract.md) exactly.
3. **No Test Set Tuning:** Improvements rely on standard linear resampling and formal hysteresis state machines; hyperparameters were not tuned to optimize test accuracy.
4. **Test Suite Integrity:** All 13 unit and integration tests pass with code 0 in under 0.45 seconds.
