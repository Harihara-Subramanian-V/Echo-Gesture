# EchoGesture: Phase 0 Project Inventory & Deliverables Matrix

**Project:** EchoGesture — Track 02: Sign Language Prediction on Edge Devices  
**Hackathon:** XR Hackathon 2026 (Samsung R&D Institute India)  
**Institution:** VIT Vellore  
**Team:** Code Directors  
**Submission Deadline:** 22 November 2026, 11:59 PM  
**Audit Timestamp:** 2026-11-21T02:20:00+05:30  
**Verification Model:** Gemini 3.8 Flash (High) ONLY  

---

## 1. Git Repository & Remote Audit

| Parameter | Local State | Remote State (`origin`) | Parity / Alignment Status |
| :--- | :--- | :--- | :--- |
| **Repository URL** | None previously configured | `https://github.com/Harihara-Subramanian-V/Echo-Gesture.git` | Configured as `origin` |
| **Current Branch** | `master` (10 commits) | `main` (1 initial commit) | Unrelated histories; to be merged cleanly before push |
| **Latest Commit** | `12cb1f4 feat(validation): Complete Phase 7E...` | `79bd930 Initial commit` (Only `# Echo-Gesture`) | Remote has minimal single-line README |
| **Remote Connectivity** | Verified via `git ls-remote` | Accessible, public repository | Ready for non-destructive merge and push |
| **Destructive Commands** | Strictly Prohibited | No force push, no history overwrite | Standard clean non-fast-forward merge strategy |

---

## 2. Deliverables Matrix (Track 02 Requirements)

| Required Deliverable | Expected Path | Purpose & Specifications | Current Status | Responsible Agent |
| :--- | :--- | :--- | :--- | :--- |
| **Track 02 Predict Interface** | `predict.py` | `predict(video_path: str) -> dict` returning `clip_id`, uppercase `prediction`, `top5`, `confidence`, `latency_ms`. Local inference only. | To be implemented and tested | **Agent 1** (Evaluator Contract) |
| **Evaluation Interface Tests** | `tests/test_predict_interface.py` | Unit tests for valid/invalid paths, empty frames, schema compliance, latency measurement. | To be implemented | **Agent 1** (Evaluator Contract) |
| **Batch Prediction Generator** | `scripts/generate_predictions_jsonl.py` | CLI tool to batch process evaluation manifests or held-out test sets. | To be implemented | **Agent 2** (Predictions & Results) |
| **Predictions JSONL** | `submission/predictions.jsonl` | 1 JSON object per line, exact clip IDs, uppercase vocabulary, actual pipeline predictions. | To be generated | **Agent 2** (Predictions & Results) |
| **Results Benchmark Table** | `submission/results.csv` | 6 official metrics: `top1_accuracy`, `latency_p95`, `model_size`, `accuracy_drop_robustness`, `avg_power`, `thermal_drift_10min`. | To be compiled with empirical evidence | **Agent 2** (Predictions & Results) |
| **Evaluation Reports** | `reports/accuracy_report.md`<br>`reports/robustness_report.md` | Formal reports for accuracy, Top-5 recovery, and perturbation robustness. | To be compiled | **Agent 2** (Predictions & Results) |
| **Demo Script & Checklist** | `submission/demo/demo_script.md`<br>`submission/demo/demo_checklist.md` | Realistic 2m 40s (max 3 min) demo video recording guide and scenario checklist. | To be drafted | **Agent 3** (Android & Demo Readiness) |
| **Power & Thermal Report** | `reports/power_thermal_report.md` | Empirical analysis of device thermals (`dumpsys battery`), power measurement constraints on Android 16. | To be compiled | **Agent 3** (Android & Demo Readiness) |
| **Android APK Verification** | `android/app/build/outputs/apk/` | Verify release/debug APK build, signature, asset inclusion, size under 20 MB. | To be audited | **Agent 3** (Android & Demo Readiness) |
| **Submission README** | `README.md` | Comprehensive GitHub frontpage: setup, architecture, dataset provenance, usage, benchmarks. | To be updated | **Agent 4** (Repo & Release) |
| **AI Disclosure Document** | `AI_DISCLOSURE.md` | Transparent, honest disclosure of AI development workflows and tools. | To be authored | **Agent 4** (Repo & Release) |
| **Model Conversion Script** | `model/model_conversion_script.py` | Exact Python code used to convert and calibrate INT8 TFLite model. | To be copied/structured | **Agent 4** (Repo & Release) |
| **Submission Model Asset** | `model/echogesture_exp_d_int8.tflite` | Packaged INT8 quantized model (135,464 bytes, SHA-256 verified). | To be tracked in git | **Agent 4** (Repo & Release) |
| **Docker Reproducibility** | `docker/Dockerfile`<br>`docker/requirements.txt` | Containerized environment for running Python evaluation and `predict.py`. | To be created | **Agent 4** (Repo & Release) |
| **Git Configuration** | `.gitignore` | Unignore INT8 model assets while strictly excluding `.venv`, caches, and credentials. | To be updated | **Agent 4** (Repo & Release) |
| **Final Readiness Report** | `reports/final_submission_readiness_report.md` | Master audit checklist, metrics table, provenance statement, blockers list, next actions. | To be synthesized | **Main Agent** |

---

## 3. Inventory of Pre-Existing Assets & Baselines

1. **Trained & Packaged Model:**
   - Architecture: Bidirectional GRU (2 layers, 128 units, dropout 0.3) + Linear Classifier.
   - Input: `[1, 64, 263]` (INT8 quantized, scale = 0.054350, zero-point = -28).
   - Output: `[1, 40]` (INT8 quantized, scale = 0.061986, zero-point = -37).
   - Footprint: **135,464 bytes (~132 KB)**. Strict target `< 20 MB` met with >99% margin.
   - Cryptographic SHA-256: `ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499`.
2. **Vocabulary:**
   - Exactly 40 discrete isolated ASL lexical signs (`data/processed/sequences_v3/label_encoder.json`).
   - Selected using WLASL metadata overlap; trained and validated exclusively on MS-ASL acquired clips. (How2Sign was NOT used).
3. **Hardware Baseline (vivo V2416):**
   - Pure Model Latency: Median **0.599 ms**, P95 **1.283 ms** (LiteRT INT8 with 2 CPU threads XNNPACK). Target `< 50 ms` met with 97% margin.
   - Vision Latency (MediaPipe): Median **80.97 ms** (asynchronously decoupled on background executor).
   - Display FPS: **59.34 - 60.0 FPS** (Display Choreographer).
   - Memory PSS: **164.5 MB - 197.6 MB** with 0 memory leak across 100+ seconds.
4. **Offline Native TTS:**
   - Android native `TextToSpeech` engine (`com.google.android.tts`, `Locale.US`), 100% offline.
5. **Existing Regression Test Suites:**
   - Android Kotlin Unit Tests: 243 / 243 passing.
   - Python Test Suite: 75 / 75 passing.
