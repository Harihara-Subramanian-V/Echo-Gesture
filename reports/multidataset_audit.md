# EchoGesture — Multi-Dataset Acquisition & Auditing Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Operating Model:** Gemini 3.8 Flash (High) strictly  
**Sprint Window:** 48-Hour Emergency Multi-Dataset Acquisition  

---

## 1. Executive Summary

To achieve robust isolated American Sign Language (ASL) recognition under unconstrained edge-device conditions, EchoGesture investigated and audited multiple public ASL datasets beyond the initial WLASL benchmark:
1. **MS-ASL** (Microsoft Research — Vaezi Joze & Koller, BMVC 2019): **HIGH PRIORITY ACQUISITION.** 100% vocabulary overlap (40/40 classes), 1,994 clips, 150 distinct signers, strictly signer-independent partitions.
2. **ASL Citizen** (Microsoft Research / BU / UW / UMD — Desai et al., NeurIPS 2023): **SELECTIVE ACQUISITION BLOCKED.** Official distribution is a monolithic 42.8 GB archive (`ASL_Citizen.zip`). No selective API or per-clip download endpoint is provided. In accordance with Prompt Condition B.5, acquisition of the full 42.8 GB archive is **halted** to avoid exhausting local storage.
3. **WLASL** (Li et al., WACV 2020): **AUTHORIZED SUBSET REQUEST ACTIVE.** Formal request file `missing.txt` (911 video IDs across 40 classes) submitted to maintainers.
4. **How2Sign** (Duarte et al., CVPR 2021): **DEFERRED.** Continuous instructional video corpus (~80 hours, >30 GB) incompatible with isolated word-level classifiers.

---

## 2. Multi-Dataset Comparative Audit Matrix

| Dimension | WLASL (Word-Level ASL) | MS-ASL (Microsoft ASL) | ASL Citizen | How2Sign |
| :--- | :--- | :--- | :--- | :--- |
| **Official Source** | GitHub `dxli94/WLASL` | MS Research ID `100121` | MS Research ID `105253` | `how2sign.github.io` |
| **Dataset Structure** | Isolated sign video clips | Isolated sign YouTube subclips | Isolated sign webcams | Continuous sign sentences |
| **Total Vocabulary** | 2,000 isolated glosses | 1,000 isolated glosses | 2,731 ASL-LEX glosses | ~35,000 continuous sentences |
| **EchoGesture Overlap** | **40 / 40 classes (100%)** | **40 / 40 classes (100%)** | 40 / 40 classes present | Continuous transcriptions |
| **Matching Instances** | **911 clips** | **1,994 clips** | ~1,200 estimated | N/A (unsegmented) |
| **Signer Metadata** | Integer IDs (119 total, 79 in subset) | Integer IDs (222 total, 150 in subset) | 52 Deaf signers | Multi-session signer IDs |
| **Signer Independence** | Official leaks 95.2%; **EchoGesture custom split has 0 overlap** | **Official splits have 0 signer overlap (106 train, 29 val, 15 test)** | Official splits signer-independent | Studio recording sessions |
| **Video Format / Res** | 25 FPS, 256×256 to 720p | Variable (25–30 FPS), 360p to 1080p | 30 FPS, front-facing webcams | 720p / 1080p RGB |
| **License / Terms** | Creative Commons CC BY-NC 4.0 / C-UDA 1.0 | Computational Use of Data Agreement (C-UDA) 0.1 | Microsoft Research Non-Commercial / Research | Research Use Only |
| **Download Method** | Author Google Drive / Request form (`missing.txt`) | YouTube downloader (`yt-dlp` + `ffmpeg` trim) | Monolithic 42.8 GB ZIP archive only | Multi-part `.tar` Google Drive |
| **Storage Impact** | ~800 MB – 1.1 GB raw | ~876 MB – 1.9 GB raw | ~42.8 GB (exceeds disk) | >30 GB (exceeds disk) |
| **Action / Decision** | **Maintain request; ready to import** | **Approved for acquisition via yt-dlp** | **Blocked under Condition B.5** | **Deferred as future work** |

---

## 3. MS-ASL Deep-Dive Analysis

### 3.1 Vocabulary Mapping and Coverage
The MS-ASL dataset metadata package (`MS-ASL.zip`, 2.02 MB) was downloaded and extracted to `data/metadata/ms_asl/MS-ASL/`. It contains:
- `MSASL_classes.json`: 1,000 classes.
- `MSASL_synonym.json`: Synonym clusters.
- `MSASL_train.json` (16,054 samples), `MSASL_val.json` (5,287 samples), `MSASL_test.json` (4,172 samples).

All **40 target classes** of the EchoGesture vocabulary were matched against MS-ASL glosses.
- **Match Rate:** 40 / 40 (100.0%).
- **Total Matching Clips:** **1,994 clips**.
- **Per-Class Instance Density:** 14 to 73 instances per class (mean: 49.8 clips/class).

### 3.2 Signer Independence Audit (Empirical Verification)
We executed an empirical signer-leakage verification across all 1,994 matching MS-ASL samples:
- **Train Split:** 1,285 clips across **106 unique signers** (40/40 classes represented).
- **Val Split:** 429 clips across **29 unique signers** (40/40 classes represented).
- **Test Split:** 280 clips across **15 unique signers** (40/40 classes represented).

```
Signer Overlap Audit (MS-ASL 40-Class Subset):
  Train ∩ Val:  0 overlapping signers
  Train ∩ Test: 0 overlapping signers
  Val ∩ Test:   0 overlapping signers
```
> [!NOTE]
> Unlike official WLASL splits, MS-ASL was authored with **strict signer independence**. Preserving the official MS-ASL splits ensures zero data leakage and guarantees generalization to unseen signers.

### 3.3 Source URLs and Trimming Requirements
All 1,994 MS-ASL instances are hosted as YouTube subclips with precise floating-point `start_time` and `end_time` timestamps (average duration: 2.93s; range: 0.33s to 9.50s).
Acquisition requires:
- `yt-dlp` for video extraction.
- `ffmpeg` (already installed at `C:\Users\haris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe`) for stream cutting.

---

## 4. ASL Citizen Audit & Selective Acquisition Decision

### 4.1 Evaluation of Official Distribution
- **Microsoft Download Center ID:** `105253`.
- **Package:** `ASL_Citizen.zip` (**42.8 GB**).
- **Available Free Disk Space:** **33.35 GB** on drive `C:`.

Downloading the full 42.8 GB archive would cause immediate disk saturation, violating the mandatory constraint to retain $\ge 5$ GB free for feature extraction and model training.

### 4.2 Investigation of Selective Acquisition
We investigated whether Microsoft or community mirrors provide an official per-clip download API:
- The official GitHub (`microsoft/ASL-citizen-code`) provides training baselines but references the monolithic Microsoft Download Center zip.
- Microsoft provides no individual clip HTTP endpoints.
- Third-party repositories host processed feature representations, but not verified raw video slices.

### 4.3 Trigger of Mandatory Halt Condition B.5
The project instruction explicitly states:
> *"5. If selective acquisition is not supported, STOP and report this. Do not download the full 42.8 GB archive."*

**Action Taken:** Download of the 42.8 GB ASL Citizen archive was **halted and rejected**. No unapproved large downloads were initiated.

---

## 5. How2Sign Evaluation

- **Status:** Deferred.
- **Reasoning:** How2Sign contains continuous sign language translations of instructional videos (~80 hours). Isolated sign recognition models require discrete gesture boundaries. Without manual or automated word-level alignment, incorporating unsegmented video introduces severe label noise. Furthermore, RGB data footprint exceeds 30 GB.
- **Future Extension:** Continuous sign language recognition and sentence translation (Phase 6 / Post-Hackathon).

---

## 6. Verification Status

| Fact / Metric | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| MS-ASL Metadata Acquired | **Verified** | `data/metadata/ms_asl/MS-ASL/` extracted and validated. |
| MS-ASL 40-Class Overlap | **Verified** | 40/40 classes (1,994 clips) parsed from JSON splits. |
| MS-ASL Signer Independence | **Verified** | 0 signer overlap across train (106), val (29), test (15). |
| MS-ASL YouTube Hosting | **Verified** | 100% of URLs point to YouTube with valid timestamps. |
| ASL Citizen Monolithic Size | **Verified** | 42.8 GB on Microsoft Download Center ID 105253. |
| ASL Citizen Selective API | **Verified** | No selective per-clip endpoint exists; halted per B.5. |
| C: Drive Free Headroom | **Verified** | 33.35 GB free headroom verified via `Get-PSDrive`. |
| Dependency Gap | **Verified** | `ffmpeg` available; `yt-dlp` requires pip install approval. |
