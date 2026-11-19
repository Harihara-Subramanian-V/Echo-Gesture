# EchoGesture — Phase 2: Storage Planning Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Status:** Storage Strategy Defined  
**Lead Model:** Gemini 3.8 Flash (High)  

---

## 1. Executive Summary

A critical constraint of the local development environment is the storage budget:
- **Available Free Disk Space:** **~39.8 GB** on drive `C:` of ~448 GB total capacity.

This report establishes a **lean, quota-conscious storage blueprint** ensuring that the end-to-end dataset acquisition, landmark caching, and model training workflow will consume **less than 3.0 GB of peak disk space**, preserving over **36.8 GB (>92%)** of free storage headroom at all times.

---

## 2. Storage Footprint Breakdown: Proposed 40-Class Pipeline

The planned EchoGesture dataset comprises **911 video clips** spanning our 40 selected sign classes.

| Processing Stage / Data Artifact | File Format | Estimated Unit Size | Total Items | Aggregate Footprint | Lifecycle & Retention Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Downloaded Video Archive** | `.zip` / `.tar.gz` | — | 1 archive | **~800 MB – 1.0 GB** | Temporary. Deleted immediately after integrity check and extraction. |
| **2. Raw Video Storage (`data/raw/`)** | `.mp4` (H.264) | ~850 KB | 911 clips | **~775 MB – 1.0 GB** | Persistent during Phase 3–4. Serves as ground truth for inspection and evaluation. |
| **3. Extracted Landmarks (`data/landmarks/`)** | `.npy` / `.npz` | ~30 KB | 911 clips | **~27.3 MB** | Persistent. Primary numeric input for feature engineering and training. |
| **4. Engineered Features (`data/features/`)** | `.npy` (float32) | ~25 KB | 911 clips | **~22.8 MB** | Persistent. Normalized, body-relative feature tensors fed into Keras DataLoader. |
| **5. Model Checkpoints (`models/checkpoints/`)** | `.keras` / `.h5` | ~5 – 15 MB | ~5 models | **~50 MB – 75 MB** | Persistent. Retains best validation checkpoints across GRU and TCNN runs. |
| **6. Quantized Models (`models/quantized/`)** | `.tflite` (INT8) | ~1.5 MB | ~3 models | **~4.5 MB** | Persistent. Final edge deployment artifacts. |
| **7. Temporary / Staging Headroom** | temp / caches | — | — | **~1.0 GB** | Ephemeral. Reserved for OS swap, OpenCV frame decoding buffers, and test logs. |
| **Total Peak Storage Footprint** | — | — | — | **~2.7 GB – 2.9 GB** | **Peak Footprint represents < 7.5% of available free disk space.** |

---

## 3. Storage Comparison: Alternatives vs. EchoGesture Strategy

| Dataset Strategy | Total Video Count | Download Size | Extracted Size | Storage Feasibility on 40 GB Workstation |
| :--- | :---: | :---: | :---: | :--- |
| **How2Sign (Sentence-Level)** | ~35,000 clips | 30 – 40 GB | 35 – 45 GB | ❌ **CRITICAL FAILURE:** Would immediately saturate or crash drive `C:`. |
| **Full WLASL-2000** | 21,083 clips | 18 – 22 GB | 22 – 26 GB | ⚠️ **HIGH RISK:** Leaves <10 GB free space; risks system OS instability during unzip. |
| **WLASL-100 Subset** | 2,038 clips | ~2.0 GB | ~2.5 GB | 🟡 **Viable but Unnecessary:** Higher class count than needed for initial edge baseline. |
| **EchoGesture 40-Class Curated** | **911 clips** | **~850 MB** | **~1.0 GB** | ✅ **OPTIMAL:** Consumes <3.0 GB total; preserves ~37 GB free space headroom. |

---

## 4. Storage Optimization Principles

To prevent unexpected disk bloat throughout Phase 2 and subsequent phases, the following rules are enforced:

### Rule 1: Targeted Subsetting
Never download or extract uncurated bulk archives. When extracting files from source mirrors, filter strictly by the 911 `video_id`s defined in our manifest.

### Rule 2: Ephemeral Archive Cleanup
Once archive integrity and video checksums are validated, the compressed archive (`.zip`) will be purged before beginning landmark extraction.

### Rule 3: Decoupled Landmark-Driven Training
Training will never read raw video files. In Phase 4, MediaPipe will extract landmark coordinates once, storing them as compact compressed NumPy arrays (`.npz` or `.npy`). Training iterations in Phase 6–7 will stream directly from the **~27 MB landmark cache**, resulting in near-instantaneous epoch loading and zero video decoding overhead during training.

### Rule 4: Automated Storage Guard
Before initiating any download or bulk extraction script, an automated pre-flight storage check will verify that $\ge 20\text{ GB}$ of free space remains on `C:`. If free space drops below $15\text{ GB}$, the pipeline will halt immediately and log an alert.

---

*This storage planning report completes Task 4 of Phase 2. No video data has been downloaded.*
